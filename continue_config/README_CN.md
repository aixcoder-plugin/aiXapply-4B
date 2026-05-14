# continue_config

这个目录提供了一套把训练好的 `Apply` 模型接入 `Continue` 的示例配置。核心思路是：

1. 将你的模型以 `OpenAI-compatible` 的 `/v1/chat/completions` 接口方式启动。
2. 使用 `continue_apply_proxy.py` 在本地做一层转发。
3. 在 `Continue` 中把这个转发地址配置成 `roles: [apply]` 的模型。

之所以需要代理，是因为训练好的 `Apply` 模型通常会输出：

```text
<update_file>...</update_file>
```

而 `Continue` 更适合直接接收“完整更新后的文件内容”。`continue_apply_proxy.py` 会在转发时自动去掉这两个标签，并兼容流式输出。

## 目录说明

- `continue.config.yaml.example`：`Continue` 的示例配置文件。
- `continue_apply_proxy.py`：本地转发服务，负责把 `Continue` 的请求转发给你的模型服务，并移除 `<update_file>` 标签。

## 适用前提

在使用这套配置前，你需要具备以下条件：

- 你的训练后模型已经可以通过 HTTP 提供 `OpenAI-compatible` 的 `chat/completions` 接口。
- 该接口能够接收 `messages` 请求，并返回完整文件内容，或返回带 `<update_file>` 包裹的完整文件内容。
- 本机已经安装并可以正常使用 `Continue`。

## 工作方式

这个目录里的示例配置只定义了 `apply` 模型。你可以保留自己现有的 `chat` 模型配置，只把 `aiXapply Apply` 这一段合并进 `Continue` 配置即可。

`Continue` 在执行 Apply 时，会把原始文件和局部修改片段拼成提示词，发送给 `apply` 模型。`apply` 模型需要返回“更新后的完整文件内容”。如果模型仍然沿用训练时的输出格式，返回了 `<update_file>...</update_file>`，代理会自动把标签剥掉，再返回给 `Continue`。

## 快速开始

### 1. 启动你的模型服务

先把训练好的模型启动成一个 `OpenAI-compatible` 服务，并确认它的接口地址类似：

```text
http://127.0.0.1:12345/v1/chat/completions
```

这里只是示例地址，你可以替换成自己的实际服务地址。

### 2. 配置并启动代理

在 `continue_config/` 目录下执行：

```bash
export APPLY_PROXY_UPSTREAM_CHAT_URL="http://127.0.0.1:12345/v1/chat/completions"
export APPLY_PROXY_HOST="127.0.0.1"
export APPLY_PROXY_PORT="14124"

python3 continue_apply_proxy.py
```

如果你的上游服务需要鉴权头，可以在启动前显式设置：

```bash
export APPLY_PROXY_UPSTREAM_AUTH="Bearer <your-token>"
python3 continue_apply_proxy.py
```

默认情况下，代理会监听：

```text
http://127.0.0.1:14124
```

可以用下面的命令检查代理是否启动成功：

```bash
curl http://127.0.0.1:14124/health
```

### 3. 配置 Continue

先备份现有的 `Continue` 配置文件。常见位置是：

```text
~/.continue/config.yaml
```

然后将 `continue.config.yaml.example` 作为模板，拷贝或合并到你的 `Continue` 配置中。

如果你已经有自己的 `chat` 模型配置，通常只需要把 `aiXapply Apply` 这一段合并进去，不一定要完全覆盖整个配置文件。

### 4. 修改示例配置中的关键字段

你至少需要检查下面几个配置项：

- `apiBase`：改成代理地址，例如 `http://127.0.0.1:14124/v1/`
- `model`：改成你的本地 Apply 模型标识
- `requestOptions.extraBodyProperties.model`：改成你上游服务实际需要的模型名
- `apiKey`：对于本地代理通常可以保留非敏感占位值，例如 `local-placeholder`

## 配置项说明

### 代理脚本环境变量

| 变量名 | 作用 | 默认值 |
| --- | --- | --- |
| `APPLY_PROXY_HOST` | 代理监听地址 | `127.0.0.1` |
| `APPLY_PROXY_PORT` | 代理监听端口 | `14124` |
| `APPLY_PROXY_UPSTREAM_CHAT_URL` | 上游模型服务地址 | `http://127.0.0.1:12345/v1/chat/completions` |
| `APPLY_PROXY_UPSTREAM_AUTH` | 转发给上游的 `Authorization` 请求头 | 空 |
| `APPLY_PROXY_HEALTH_SHOW_UPSTREAM` | 设置为 `1`、`true` 或 `yes` 时，在 `/health` 输出里包含上游 URL | 空 |

### `continue.config.yaml.example` 中的关键字段

- `roles: [apply]`：告诉 `Continue` 这个模型专门用于 Apply。
- `promptTemplates.apply`：自定义 Apply 提示词模板。
- `useLegacyCompletionsEndpoint: false`：明确走 `chat/completions` 而不是旧接口。
- `defaultCompletionOptions.contextLength`：Apply 请求允许的上下文长度。
- `defaultCompletionOptions.maxTokens`：生成上限。
- `defaultCompletionOptions.temperature: 0.3`：尽量降低不确定性输出。
- `requestOptions.extraBodyProperties.model`：显式在请求体里传递上游模型标识；如果你的服务依赖特定模型 ID，请修改这里。

## 推荐接入方式

如果你的训练模型本来就是用下面这类格式训练的：

```text
<source_file>...</source_file>
<update_snippet>...</update_snippet>
<update_file>...</update_file>
```

那么推荐继续保留当前这套“`Continue -> 本地代理 -> 模型服务`”的结构，原因有两个：

1. `Continue` 侧可以直接拿到干净的完整文件内容。
2. 即使模型偶尔输出 `<update_file>` 标签，代理也能自动兼容。

## 常见问题

### 1. Continue 能连上，但 Apply 没有结果

优先检查：

- 你的上游服务是否真的实现了 `/v1/chat/completions`
- `APPLY_PROXY_UPSTREAM_CHAT_URL` 是否写成了完整路径，而不只是 `/v1`
- 上游服务是否要求真实的 `Authorization` 请求头
- 上游服务要求的模型名是否已经填到 `requestOptions.extraBodyProperties.model`

### 2. 模型输出里还带 `<update_file>` 标签

这是正常现象。代理会在返回给 `Continue` 之前自动移除。

### 3. 我已经有自己的 Continue 配置，是否必须整体替换

不需要。最常见的做法是：

- 保留你现有的 `chat` / `autocomplete` / 其他模型配置
- 只把示例中的 `apply` 模型块追加进去
- 按你的本地端口和模型名调整 `apiBase` 与 `requestOptions.extraBodyProperties.model`

### 4. 我的模型已经直接输出纯代码，还需要代理吗

通常仍然建议保留代理。这样可以统一处理流式返回和标签兼容问题，减少 `Continue` 侧配置差异。

## 最小接入清单

完成以下 4 步后，通常就可以在 `Continue` 中使用你的训练后模型了：

1. 启动一个 `OpenAI-compatible` 的模型服务。
2. 启动 `continue_apply_proxy.py`。
3. 将 `continue.config.yaml.example` 合并到 `Continue` 配置。
4. 把 `apiBase`、上游模型名和鉴权信息改成你的实际值。
