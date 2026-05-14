# continue_config

This directory provides an example setup for using a trained `Apply` model inside `Continue`. The basic workflow is:

1. Serve your model through an OpenAI-compatible `/v1/chat/completions` endpoint.
2. Put `continue_apply_proxy.py` in front of that model as a local proxy.
3. Configure `Continue` to use the proxy as a model with `roles: [apply]`.

The proxy is needed because trained `Apply` models often return outputs wrapped like:

```text
<update_file>...</update_file>
```

`Continue` works better when it directly receives the final updated file content. `continue_apply_proxy.py` removes these tags before returning the response, and it also supports streaming responses.

## Files In This Directory

- `continue.config.yaml.example`: example `Continue` config.
- `continue_apply_proxy.py`: local forwarding proxy that sends requests from `Continue` to your model service and strips `<update_file>` tags from the response.
- `README_CN.md`: Chinese version of this document.

## Prerequisites

Before using this setup, make sure:

- Your trained model is exposed through an HTTP endpoint compatible with OpenAI `chat/completions`.
- That endpoint accepts `messages` and returns the full updated file, or the full updated file wrapped in `<update_file>...</update_file>`.
- `Continue` is already installed and working on your machine.

## How It Works

The sample configuration in this directory only defines the `apply` model. You can keep your own existing `chat` model entries in `Continue` and simply merge the `aiXapply Apply` block into your config.

When `Continue` performs an Apply action, it sends the original file content plus the edit snippet to the `apply` model. The `apply` model is expected to return the full updated file. If the model still uses the training-time output format and returns `<update_file>...</update_file>`, the proxy strips those tags before sending the result back to `Continue`.

## Quick Start

### 1. Start your model service

Serve your trained model through an OpenAI-compatible endpoint, for example:

```text
http://127.0.0.1:12345/v1/chat/completions
```

This is only an example. Replace it with your real upstream endpoint.

### 2. Configure and start the proxy

Inside `continue_config/`, run:

```bash
export APPLY_PROXY_UPSTREAM_CHAT_URL="http://127.0.0.1:12345/v1/chat/completions"
export APPLY_PROXY_HOST="127.0.0.1"
export APPLY_PROXY_PORT="14124"

python3 continue_apply_proxy.py
```

If your upstream service requires an authentication header, set it explicitly before starting the proxy:

```bash
export APPLY_PROXY_UPSTREAM_AUTH="Bearer <your-token>"
python3 continue_apply_proxy.py
```

By default, the proxy listens on:

```text
http://127.0.0.1:14124
```

You can verify that the proxy is running with:

```bash
curl http://127.0.0.1:14124/health
```

### 3. Configure Continue

Back up your existing `Continue` config first. A common location is:

```text
~/.continue/config.yaml
```

Then use `continue.config.yaml.example` as a template and copy or merge it into your `Continue` config.

If you already have your own `chat` model configuration, you usually only need to merge the `aiXapply Apply` block instead of replacing the whole file.

### 4. Update the key fields in the sample config

At minimum, review these fields:

- `apiBase`: point this to the proxy, for example `http://127.0.0.1:14124/v1/`
- `model`: set this to a stable identifier for your local Apply model
- `requestOptions.extraBodyProperties.model`: change this to the actual model name expected by your upstream service
- `apiKey`: for a local proxy this can usually stay as a non-secret placeholder value, such as `local-placeholder`

## Configuration Reference

### Proxy environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `APPLY_PROXY_HOST` | Proxy listening host | `127.0.0.1` |
| `APPLY_PROXY_PORT` | Proxy listening port | `14124` |
| `APPLY_PROXY_UPSTREAM_CHAT_URL` | Upstream model service URL | `http://127.0.0.1:12345/v1/chat/completions` |
| `APPLY_PROXY_UPSTREAM_AUTH` | `Authorization` header forwarded to the upstream service | empty |
| `APPLY_PROXY_HEALTH_SHOW_UPSTREAM` | Include the upstream URL in `/health` output when set to `1`, `true`, or `yes` | empty |

### Important fields in `continue.config.yaml.example`

- `roles: [apply]`: tells `Continue` that this model is dedicated to Apply actions.
- `promptTemplates.apply`: custom prompt template for Apply.
- `useLegacyCompletionsEndpoint: false`: explicitly uses `chat/completions` instead of the legacy endpoint.
- `defaultCompletionOptions.contextLength`: context length allowed for Apply requests.
- `defaultCompletionOptions.maxTokens`: generation limit.
- `defaultCompletionOptions.temperature: 0.3`: keeps output more deterministic.
- `requestOptions.extraBodyProperties.model`: explicitly passes the upstream model identifier in the request body; update this if your service depends on a specific model ID.

## Recommended Integration Pattern

If your trained model was built around a format like:

```text
<source_file>...</source_file>
<update_snippet>...</update_snippet>
<update_file>...</update_file>
```

then it is recommended to keep the current structure:

```text
Continue -> local proxy -> model service
```

This has two main benefits:

1. `Continue` receives clean full-file output directly.
2. The proxy still handles `<update_file>` tags automatically if the model emits them.

## Common Issues

### 1. Continue can connect, but Apply returns nothing

Check the following first:

- Your upstream service really implements `/v1/chat/completions`
- `APPLY_PROXY_UPSTREAM_CHAT_URL` is the full endpoint path, not just `/v1`
- Your upstream service requires a real `Authorization` header
- The model name required by the upstream service is correctly set in `requestOptions.extraBodyProperties.model`

### 2. The model output still contains `<update_file>` tags

That is expected. The proxy removes them before returning the response to `Continue`.

### 3. I already have my own Continue config. Do I need to replace all of it?

No. The most common approach is:

- Keep your existing `chat`, `autocomplete`, or other model entries
- Add only the sample `apply` model block
- Adjust `apiBase` and `requestOptions.extraBodyProperties.model` to match your own local port and model name

### 4. My model already returns plain code. Do I still need the proxy?

In most cases, keeping the proxy is still recommended. It gives you a consistent integration layer for streaming responses and tag compatibility.

## Minimal Checklist

Once you finish these four steps, your trained model should be usable in `Continue`:

1. Start an OpenAI-compatible model service.
2. Start `continue_apply_proxy.py`.
3. Merge `continue.config.yaml.example` into your `Continue` config.
4. Update `apiBase`, upstream model name, and authentication settings to your real values.
