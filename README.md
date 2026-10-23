# `@nexustools/aitooling`

**`@nexustools/aitooling`** is a comprehensive TypeScript/ESM framework for building multi-provider agentic AI systems, context-gated tool registries, live-streaming partial-JSON tool parsers, and real-time AI chat UIs with collapsible reasoning and live tool call blocks.

[![npm version](https://img.shields.io/npm/v/@nexustools/aitooling/beta.svg)](https://www.npmjs.com/package/@nexustools/aitooling)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Key Capabilities

1. **Tool & Context-Gated Tool Registry (`Tool`, `AbstractAiTool`, `ContextualToolRegistry`)**:
   - Define structured tools with OpenAI/JSON-Schema parameters or prompt-based `@tool_name{...}` invocation.
   - Register context-specific tools that dynamically unlock or hide based on runtime state—for example, `AICore` tools that are only available after an implementation plan or task list is created (`requiresPlanOrTaskList: true`, `hideAfterPlanOrTaskList: true`, `enforcePlanGate: true`, or custom `when(context)` predicates).
2. **Multi-Provider AI Abstraction (`ProviderFactory`, `LmStudioAiClient` / `MultiProviderToolStreamClient`)**:
   - Unified streaming and tool-calling interface across:
     - **OpenAI** (`gpt-4o`, `gpt-4o-mini`, `o3-mini`, `gpt-4-turbo`)
     - **Anthropic Claude** (`claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`)
     - **Google Gemini** (`gemini-2.5-pro`, `gemini-2.5-flash` via API Key or **Google One OAuth** via `GoogleOAuthManager`)
     - **xAI Grok** (`grok-2-latest`, `grok-2-1212`, `grok-beta`)
     - **LM Studio** (`http://127.0.0.1:1234` local/remote OpenAI-compatible server with `/api/v1/models` and `/v1/models` auto-discovery)
     - **Custom OpenAI / Custom Claude Endpoints**
3. **Custom AI Service Interfaces via Code (`CustomAiServiceAdapter`, `AiProviderRegistry`, `ProviderFactory.registerCustomProvider`)**:
   - Projects using `@nexustools/aitooling` can implement and register their own custom AI service providers programmatically and select them anywhere a built-in provider is used.
4. **Live Streaming Partial-JSON & Thought Parser (`parsePartialJson`, `ToolParser`, `extractLiveThinkingAndReply`)**:
   - Parses incomplete streaming JSON arguments in real time so UIs and canvases can preview tool parameters token-by-token before the LLM finishes generating the tool call.
   - Extracts `<think>...</think>` and `<thinking>...</thinking>` blocks live during streaming.
5. **Live AI Chat UI Controller (`AiLiveChatController`)**:
   - Reusable DOM controller that renders streaming assistant replies, collapsible live reasoning blocks with configurable thinking budget (`none`, `low`, `medium`, `max`), and collapsible live-updating tool call parameter cards.
6. **Model Catalog, Cost Calculator, & Engine Queue (`ModelCatalog`, `CostCalculator`, `EngineQueue`, `ConfigManager`)**:
   - Pre-populated model database with parameter counts (`8B`, `70B`, `685B`, etc.), vision support flags, live SDK model discovery, token cost calculation (`$/1M` input/output tokens), and serialized/concurrent execution queues.

---

## Installation

```bash
npm install @nexustools/aitooling@beta
```

---

## Quick Start

### 1. Creating Tools & Using `ContextualToolRegistry`

```typescript
import {
    AbstractAiTool,
    ContextualToolRegistry,
    AiToolDefinition,
    AiToolExecutionResult
} from "@nexustools/aitooling";

interface AppAgentContext {
    hasPlan: boolean;
    taskList: string[];
    workspaceDir: string;
}

class CreatePlanTool extends AbstractAiTool<AppAgentContext> {
    readonly name = "create_plan";
    readonly description = "Create the implementation plan before writing code.";
    protected readonly parameters: AiToolDefinition["function"]["parameters"] = {
        type: "object",
        properties: {
            tasks: { type: "array", items: { type: "string" }, description: "Ordered list of tasks" }
        },
        required: ["tasks"]
    };

    async execute(args: Record<string, any>, context: AppAgentContext): Promise<AiToolExecutionResult> {
        context.taskList = Array.isArray(args.tasks) ? args.tasks : [];
        context.hasPlan = context.taskList.length > 0;
        return { ok: true, summary: `Created plan with ${context.taskList.length} tasks.` };
    }
}

class EditFilesTool extends AbstractAiTool<AppAgentContext> {
    readonly name = "edit_files";
    readonly description = "Edit project source files (only unlocked after a plan is created).";
    protected readonly parameters: AiToolDefinition["function"]["parameters"] = {
        type: "object",
        properties: {
            path: { type: "string" },
            content: { type: "string" }
        },
        required: ["path", "content"]
    };

    async execute(args: Record<string, any>, context: AppAgentContext): Promise<AiToolExecutionResult> {
        return { ok: true, summary: `Edited ${args.path} in ${context.workspaceDir}` };
    }
}

const registry = new ContextualToolRegistry<AppAgentContext>();

// Available before a plan exists:
registry.register(new CreatePlanTool(), { hideAfterPlanOrTaskList: true });

// Only unlocked AFTER a plan or task list is created:
registry.register(new EditFilesTool(), {
    requiresPlanOrTaskList: true,
    enforcePlanGate: true
});
```

### 2. Implementing a Custom AI Service Provider via Code

Projects can register their own AI service providers using `AiProviderRegistry.registerProvider()` (for the multi-turn tool stream client) or `ProviderFactory.registerCustomProvider()` (for `BaseAIProvider` instances):

```typescript
import {
    AiProviderRegistry,
    CustomAiServiceAdapter,
    LmStudioAiClient
} from "@nexustools/aitooling";

const myCustomAdapter: CustomAiServiceAdapter = {
    id: "my-company-llm",
    displayName: "My Company Internal LLM",
    async listModels(config) {
        return [{ id: "company-coder-70b", object: "model", owned_by: "internal" }];
    },
    async streamChatTurn({ messages, tools, callbacks, abortSignal, config }) {
        callbacks?.onContentDelta?.("Hello from custom provider!");
        return {
            assistantMessage: {
                role: "assistant",
                content: "Hello from custom provider!"
            }
        };
    }
};

// Register globally so any LmStudioAiClient / MultiProviderToolStreamClient can use it:
AiProviderRegistry.registerProvider(myCustomAdapter);

const client = new LmStudioAiClient({
    provider: "my-company-llm",
    model: "company-coder-70b"
});
```

### 3. Live-Parsing AI Chat UI (`AiLiveChatController`)

```typescript
import { AiLiveChatController, LmStudioAiClient } from "@nexustools/aitooling";

const client = new LmStudioAiClient({
    provider: "openai",
    apiKey: process.env.OPENAI_API_KEY,
    model: "gpt-4o"
});

const chatUi = new AiLiveChatController({
    messagesContainer: document.getElementById("chat-messages")!,
    statusBadgeEl: document.getElementById("ai-status")!,
    sendButtonEl: document.getElementById("btn-send") as HTMLButtonElement,
    stopButtonEl: document.getElementById("btn-stop") as HTMLButtonElement,
    client
});

await chatUi.sendMessage("Create a plan and build the login component.");
```

---

## Documentation

See [docs/ARCHITECTURE_AND_GUIDE.md](docs/ARCHITECTURE_AND_GUIDE.md) for the complete architectural guide and API reference.
