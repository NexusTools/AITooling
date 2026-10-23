# `@nexustools/aitooling` — Architecture & Developer Guide

## Overview

`@nexustools/aitooling` provides the core AI infrastructure shared across **AICore**, **HACanvas**, **HAEditor**, and third-party applications:

```
@nexustools/aitooling
├── tools/        Tool, AbstractAiTool, ContextualToolRegistry (context-gated tools)
├── parser/       parsePartialJson, ToolParser, extractLiveThinkingAndReply
├── providers/    BaseAIProvider, OpenAI/Claude/Gemini/Grok/LMStudio, CustomAiServiceAdapter,
│                 AiProviderRegistry, AbstractAiStreamClient, LmStudioAiClient
├── catalog/      ModelCatalog (parameter complexity, vision detection, live discovery)
├── cost/         CostCalculator ($/1M input/output token pricing & custom rates)
├── config/       ConfigManager (encrypted key storage, multi-provider AppConfig)
├── auth/         GoogleOAuthManager (Google One OAuth PKCE flow for Gemini)
├── engine/       EngineQueue (serialized & concurrent task execution queue)
└── ui/           AiLiveChatController (live-parsing streaming chat + tool call UI)
```

---

## 1. Context-Specific & Plan-Gated Tools (`ContextualToolRegistry`)

Complex agentic applications such as **AICore** require tools whose availability changes dynamically based on the agent's current state (for instance, preventing an agent from editing files or spawning workers until an implementation plan or task list has been created).

`ContextualToolRegistry<TContext>` supports four declarative rules per tool in `ToolContextRule<TContext>`:

| Rule Property | Type | Description |
|---|---|---|
| `when` | `(context: TContext) => boolean` | Arbitrary predicate evaluated dynamically whenever `getAvailableTools(context)` or `executeTool(...)` is called. |
| `requiresPlanOrTaskList` | `boolean` | When `true`, the tool is hidden from the LLM's tool definitions and blocked from execution until the context has an active plan or non-empty task list. |
| `hideAfterPlanOrTaskList` | `boolean` | When `true`, the tool is only exposed before a plan/task list exists and automatically hidden afterwards. |
| `enforcePlanGate` | `boolean` | When `true`, attempting to execute the tool before a plan exists returns a structured error prompting the model to create its plan or task list first. |

### Live Partial-JSON Streaming Preview (`onStreamArgsPreview`)

Every `AbstractAiTool<TContext>` can optionally implement `onStreamArgsPreview(partialArgs, context)`. While the model is still streaming a tool call's JSON arguments token-by-token, `LmStudioAiClient` parses the incomplete JSON via `parsePartialJson()` and invokes `onStreamArgsPreview` so applications (like `HACanvas` / `HAEditor`) can preview shapes, strokes, or edits in real time.

---

## 2. Multi-Provider AI Support

`@nexustools/aitooling` supports all major AI providers out of the box, both through the message-based `ProviderFactory` (`BaseAIProvider`) and the OpenAI-tool-schema-compatible `LmStudioAiClient` (`MultiProviderToolStreamClient`):

- **`lmstudio`**: Local or remote LM Studio server (`http://127.0.0.1:1234`), supporting both `/v1/models` and `/api/v1/models` discovery and SSE tool streaming.
- **`openai`**: OpenAI official API (`https://api.openai.com/v1`).
- **`claude`**: Anthropic Messages API (`https://api.anthropic.com/v1/messages`) with automatic translation between OpenAI `tools`/`tool_calls` and Anthropic `tool_use`/`tool_result` content blocks.
- **`gemini`** / **`gemini-googleone`**: Google Gemini (`https://generativelanguage.googleapis.com/v1beta/openai`) using either a Gemini API Key or Google One OAuth bearer token via `GoogleOAuthManager`.
- **`grok`**: xAI Grok API (`https://api.x.ai/v1`).
- **`custom-openai`**: Any custom OpenAI-compatible endpoint (`baseUrl`, `apiKey`, `model`).
- **`custom-claude`**: Any custom Anthropic Claude-compatible endpoint (`baseUrl`, `apiKey`, `model`).

---

## 3. Creating Custom AI Service Interfaces via Code

Applications that need to connect to proprietary LLM gateways, local inference engines, or custom transports (WebSockets, gRPC, IPC) can implement their own provider in code in two ways:

### A. Registering a `CustomAiServiceAdapter` with `AiProviderRegistry`

`AiProviderRegistry.registerProvider(adapter)` registers a custom provider for `LmStudioAiClient` / `MultiProviderToolStreamClient` and `AiLiveChatController`:

```typescript
import {
    AiProviderRegistry,
    CustomAiServiceAdapter,
    LmStudioAiClient
} from "@nexustools/aitooling";

const customAdapter: CustomAiServiceAdapter = {
    id: "custom-gateway",
    displayName: "Enterprise AI Gateway",
    async listModels(config) {
        return [
            { id: "enterprise-sonnet", object: "model", owned_by: "enterprise" }
        ];
    },
    async streamChatTurn({ messages, tools, temperature, maxTokens, callbacks, abortSignal, config }) {
        // Stream from your custom backend and fire callbacks:
        callbacks?.onThinkingDelta?.("Analyzing request...");
        callbacks?.onContentDelta?.("Completed analysis.");
        return {
            assistantMessage: {
                role: "assistant",
                content: "Completed analysis."
            }
        };
    }
};

AiProviderRegistry.registerProvider(customAdapter);
```

You can also pass `customAdapter` directly in `LmStudioClientOptions` without global registration:

```typescript
const client = new LmStudioAiClient({
    provider: "custom-gateway",
    customAdapter
});
```

### B. Registering a `BaseAIProvider` Factory with `ProviderFactory`

For applications using `ProviderFactory.createProvider(config)` (like `AICore`):

```typescript
import { BaseAIProvider, ProviderFactory, ProviderMessage } from "@nexustools/aitooling";

class MyCustomProvider extends BaseAIProvider {
    constructor(model: string) {
        super(model);
    }

    async generateResponse(
        systemPrompt: string,
        messages: ProviderMessage[],
        onStream?: (chunk: string) => void,
        onThoughtStream?: (chunk: string) => void
    ) {
        if (onStream) onStream("Response from MyCustomProvider");
        return {
            text: "Response from MyCustomProvider",
            usage: { inputTokens: 100, outputTokens: 25 }
        };
    }
}

ProviderFactory.registerCustomProvider("my-custom-provider", (config, overrideModel) => {
    return new MyCustomProvider(overrideModel || "default-model");
});
```

---

## 4. Live AI Chat UI (`AiLiveChatController`)

`AiLiveChatController` provides a drop-in, zero-framework DOM controller that renders:
- User messages and live-streaming assistant markdown/text replies
- Live-updating collapsible `💭 Thinking...` `<details>` blocks with selectable thinking mode (`none`, `low`, `medium`, `max`)
- Live-updating collapsible `🔧 tool_name` `<details>` blocks that display partial JSON arguments as they stream and update with `✓` / `✗` execution results and optional `Snapshot Preview` images.
