import { parsePartialJson } from "../parser/index.js";
import type { AbstractAiTool, ContextualToolRegistry } from "../tools/index.js";
import {
    AiProviderRegistry,
    type AbstractAiStreamClient,
    type AiChatMessage,
    type AiReasoningLevel,
    type AiReasoningMode
} from "../providers/index.js";
import type { ProviderType } from "../config/index.js";

export interface StreamingToolCallEntry {
    index: number;
    id: string;
    name: string;
    argumentsStr: string;
    bubbleEl: HTMLElement;
    statusLabelEl: HTMLElement;
    argsPreviewEl: HTMLElement;
}

export interface QuickPromptChip {
    label: string;
    prompt: string;
    title: string;
}

export interface AiLiveChatControllerOptions<TContext = any> {
    doc?: Document;
    state: TContext;
    aiClient: AbstractAiStreamClient<any>;
    toolRegistry: {
        getAllTools(): AbstractAiTool<TContext>[];
        getTool(name: string): AbstractAiTool<TContext> | undefined;
        getLiveToolStatusLabel(name: string, partialArgs: Record<string, any>, rawArgsStr: string): string;
        getCompletedToolStatusLabel(name: string, args: Record<string, any>, result: any): string;
    } | ContextualToolRegistry<TContext>;
    starterPrompts?: QuickPromptChip[];
    welcomeTitle?: string;
    inputPlaceholder?: string;
    defaultReasoningLevel?: AiReasoningLevel;
    onReasoningLevelChange?: (level: AiReasoningLevel) => void;
    onAfterToolExecuted?: (toolName: string, args: Record<string, any>, result: any) => void;
    onStatusChange?: (status: string) => void;
}

/**
 * Reusable live-parsing & live-rendering AI Chat UI controller provided by `@nexustools/aitooling`.
 *
 * Mounts into any DOM container and renders:
 * - Multi-provider connection badge (`LM Studio`, `OpenAI`, `Claude`, `Gemini`, `Grok`, `Custom`, or custom code providers)
 * - Starter prompt chips
 * - Scrollable chat transcript with live reasoning (`<think>` & `reasoning_content`), live partial-JSON
 *   tool argument pills (`parsePartialJson`), visual snapshot previews, and markdown formatting
 * - Fast Send (`➡️`, non-reasoning `"none"`) button + Brain (`🧠`) 3-level reasoning dropdown (`low`, `medium`, `max`)
 */
export class AiLiveChatController<TContext = any> {
    public readonly state: TContext;
    public readonly aiClient: AbstractAiStreamClient<any>;
    public readonly toolRegistry: AiLiveChatControllerOptions<TContext>["toolRegistry"];
    private readonly options: AiLiveChatControllerOptions<TContext>;

    public messages: AiChatMessage[] = [];
    public selectedReasoningLevel: AiReasoningLevel;
    public lastUsedReasoningMode: AiReasoningMode = "none";
    public isStreaming = false;
    private activeAbortController: AbortController | null = null;
    private isDropdownOpen = false;

    public headerBadgeEl!: HTMLElement;
    public chatContainerEl!: HTMLElement;
    public inputEl!: HTMLInputElement;
    public sendBtnEl!: HTMLButtonElement;
    public brainBtnEl!: HTMLButtonElement;
    public brainDropdownEl!: HTMLElement;

    constructor(options: AiLiveChatControllerOptions<TContext>) {
        this.options = options;
        this.state = options.state;
        this.aiClient = options.aiClient;
        this.toolRegistry = options.toolRegistry;
        this.selectedReasoningLevel = options.defaultReasoningLevel || "medium";
    }

    private get doc(): Document {
        return this.options.doc || document;
    }

    private createElement<K extends keyof HTMLElementTagNameMap>(
        tag: K,
        className?: string,
        textContent?: string,
        attrs?: Record<string, string>
    ): HTMLElementTagNameMap[K] {
        const el = this.doc.createElement(tag);
        if (className) el.className = className;
        if (textContent !== undefined) el.textContent = textContent;
        if (attrs) {
            for (const [k, v] of Object.entries(attrs)) {
                el.setAttribute(k, v);
            }
        }
        return el;
    }

    private getAllAiTools(): AbstractAiTool<TContext>[] {
        if ("getAllAiTools" in this.toolRegistry && typeof this.toolRegistry.getAllAiTools === "function") {
            return this.toolRegistry.getAllAiTools();
        }
        if ("getAllTools" in this.toolRegistry) {
            return (this.toolRegistry as any).getAllTools();
        }
        return [];
    }

    private lookupAiTool(name: string): AbstractAiTool<TContext> | undefined {
        if ("getAiTool" in this.toolRegistry && typeof this.toolRegistry.getAiTool === "function") {
            return this.toolRegistry.getAiTool(name);
        }
        if ("getTool" in this.toolRegistry) {
            return (this.toolRegistry as any).getTool(name);
        }
        return undefined;
    }

    public getConnectionBadgeText(): string {
        const provider = this.aiClient.getActiveProvider();
        const providerName = this.aiClient.getProviderDisplayName();
        const toolCount = this.getAllAiTools().length;
        if (!provider || provider === "lmstudio" || provider === "custom-openai" || provider === "custom-claude") {
            return `🔌 ${providerName} (${this.aiClient.getBaseUrl()}) • ${toolCount} Tools Ready`;
        }
        return `🔌 ${providerName} (${this.aiClient.getModelName()}) • ${toolCount} Tools Ready`;
    }

    public updateHeaderBadge(): void {
        if (this.headerBadgeEl) {
            this.headerBadgeEl.textContent = this.getConnectionBadgeText();
        }
        this.updateBrainButtonLabel();
    }

    public mount(bodyEl: HTMLElement): void {
        const wrapper = this.createElement("div", "ha-ai-panel-wrapper");

        // 1. Connection & Status Bar
        const statusRow = this.createElement("div", "ha-ai-connection-bar");
        this.headerBadgeEl = this.createElement(
            "span",
            "ha-ai-conn-badge",
            this.getConnectionBadgeText()
        );
        statusRow.appendChild(this.headerBadgeEl);
        wrapper.appendChild(statusRow);

        // 2. Starter Prompt Chips
        const chips = this.options.starterPrompts || [];
        if (chips.length > 0) {
            const chipsRow = this.createElement("div", "ha-ai-quick-chips");
            for (const item of chips) {
                const chip = this.createElement("button", "ha-ai-chip-btn", item.label, {
                    type: "button",
                    title: item.title
                });
                chip.addEventListener("click", () => {
                    this.inputEl.value = item.prompt;
                    this.inputEl.focus();
                });
                chipsRow.appendChild(chip);
            }
            wrapper.appendChild(chipsRow);
        }

        // 3. Scrollable Chat Transcript Container
        this.chatContainerEl = this.createElement("div", "ha-ai-chat-container", undefined, {
            id: "ha-ai-chat-container"
        });
        this.appendWelcomeBubble();
        wrapper.appendChild(this.chatContainerEl);

        // 4. Input Footer with Fast Send (`➡️`) and Brain Button (`🧠`) with 3-Level Dropdown (`low`, `medium`, `max`)
        const footerEl = this.createElement("div", "ha-ai-chat-footer");

        this.inputEl = this.createElement("input", "ha-input ha-ai-chat-input", undefined, {
            id: "ha-ai-chat-input",
            type: "text",
            placeholder:
                this.options.inputPlaceholder ||
                "Ask AI to inspect, draw vectors, paint, filter, or animate..."
        });

        this.inputEl.addEventListener("keydown", (e: KeyboardEvent) => {
            if (e.key === "Enter") {
                e.preventDefault();
                if (this.isStreaming) return;
                if (e.shiftKey) {
                    void this.sendMessage(this.selectedReasoningLevel);
                } else {
                    void this.sendMessage("none");
                }
            }
        });

        this.sendBtnEl = this.createElement("button", "ha-btn ha-btn-primary ha-ai-send-btn", "➡️", {
            id: "ha-ai-send-btn",
            type: "button",
            title: "Fast Send (Non-Reasoning Mode — Enter)"
        });
        this.sendBtnEl.addEventListener("click", () => {
            if (this.isStreaming) {
                this.abortActiveStream();
            } else {
                void this.sendMessage("none");
            }
        });

        const brainWrap = this.createElement("div", "ha-ai-brain-wrap");
        this.brainBtnEl = this.createElement("button", "ha-btn ha-btn-secondary ha-ai-brain-btn", "", {
            id: "ha-ai-brain-btn",
            type: "button",
            title: "Click to choose Reasoning Level (Low, Medium, Max) & Send with Reasoning"
        });
        this.updateBrainButtonLabel();

        this.brainDropdownEl = this.createElement("div", "ha-ai-brain-dropdown", undefined, {
            id: "ha-ai-brain-dropdown"
        });
        this.brainDropdownEl.style.display = "none";
        this.renderBrainDropdownItems();

        this.brainBtnEl.addEventListener("click", (e) => {
            e.stopPropagation();
            if (this.isStreaming) {
                this.abortActiveStream();
                return;
            }
            this.toggleBrainDropdown();
        });

        this.doc.addEventListener("click", (e) => {
            if (this.isDropdownOpen && !brainWrap.contains(e.target as Node)) {
                this.closeBrainDropdown();
            }
        });

        brainWrap.append(this.brainBtnEl, this.brainDropdownEl);
        footerEl.append(this.inputEl, this.sendBtnEl, brainWrap);
        wrapper.appendChild(footerEl);

        bodyEl.appendChild(wrapper);
    }

    public updateBrainButtonLabel(): void {
        if (!this.brainBtnEl) return;
        if (this.isStreaming) {
            this.brainBtnEl.textContent = "⏹";
            this.brainBtnEl.title = "Stop AI Generation";
            return;
        }
        const shortLabel =
            this.selectedReasoningLevel === "low"
                ? "Low"
                : this.selectedReasoningLevel === "medium"
                  ? "Med"
                  : "Max";
        this.brainBtnEl.innerHTML = `🧠 <span class="ha-ai-brain-level-tag">${shortLabel}</span> ▾`;
    }

    public toggleBrainDropdown(): void {
        this.isDropdownOpen = !this.isDropdownOpen;
        this.brainDropdownEl.style.display = this.isDropdownOpen ? "flex" : "none";
        if (this.isDropdownOpen) {
            this.renderBrainDropdownItems();
        }
    }

    public closeBrainDropdown(): void {
        this.isDropdownOpen = false;
        if (this.brainDropdownEl) {
            this.brainDropdownEl.style.display = "none";
        }
    }

    private renderBrainDropdownItems(): void {
        if (!this.brainDropdownEl) return;
        this.brainDropdownEl.innerHTML = "";

        const header = this.createElement(
            "div",
            "ha-ai-brain-dropdown-title",
            "🧠 Reasoning Mode Level (Click to Select & Send)"
        );
        this.brainDropdownEl.appendChild(header);

        const levels: Array<{ level: AiReasoningLevel; title: string; desc: string; badge: string }> = [
            {
                level: "low",
                title: "Low Reasoning",
                desc: "Quick thinking for fast single-tool edits & inspections",
                badge: "LOW"
            },
            {
                level: "medium",
                title: "Medium Reasoning",
                desc: "Balanced multi-step planning for layers, vectors & filters",
                badge: "MED"
            },
            {
                level: "max",
                title: "Max Reasoning",
                desc: "Deepest artistic & architectural thinking for complex scenes",
                badge: "MAX"
            }
        ];

        for (const item of levels) {
            const isSelected = this.selectedReasoningLevel === item.level;
            const optBtn = this.createElement(
                "button",
                `ha-ai-brain-option ${isSelected ? "active" : ""}`,
                undefined,
                {
                    id: `ha-ai-reasoning-${item.level}`,
                    type: "button",
                    "data-reasoning-level": item.level
                }
            );
            optBtn.innerHTML = `
                <div class="ha-ai-brain-opt-top">
                    <span class="ha-ai-brain-opt-name">${isSelected ? "✓ " : ""}${item.title}</span>
                    <span class="ha-ai-brain-opt-badge level-${item.level}">${item.badge}</span>
                </div>
                <div class="ha-ai-brain-opt-desc">${item.desc}</div>
            `;
            optBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                this.selectReasoningLevel(item.level, true);
            });
            this.brainDropdownEl.appendChild(optBtn);
        }
    }

    public selectReasoningLevel(level: AiReasoningLevel, sendIfInputPresent = true): void {
        this.selectedReasoningLevel = level;
        this.options.onReasoningLevelChange?.(level);
        this.updateBrainButtonLabel();
        this.closeBrainDropdown();

        if (sendIfInputPresent && this.inputEl && this.inputEl.value.trim().length > 0 && !this.isStreaming) {
            void this.sendMessage(level);
        }
    }

    public appendWelcomeBubble(): void {
        if (!this.chatContainerEl) return;
        const bubble = this.createElement("div", "ha-ai-bubble ha-ai-bubble-assistant ha-ai-welcome-bubble");
        const providerName = this.aiClient.getProviderDisplayName();
        bubble.innerHTML = `
            <div class="ha-ai-bubble-title">${this.options.welcomeTitle || "🤖 HAEditor AI Studio Assistant"}</div>
            <div class="ha-ai-bubble-text">
                Connected to <b>${providerName}</b> (<code>${this.aiClient.getBaseUrl()}</code>).<br/>
                • Click <b>⚙️</b> to switch between <b>LM Studio</b>, <b>OpenAI</b>, <b>Claude</b>, <b>Gemini</b>, <b>Grok</b>, or custom endpoints.<br/>
                • Click <b>➡️</b> (or press <code>Enter</code>) for <b>Fast Non-Reasoning</b> tool execution.<br/>
                • Click <b>🧠</b> to pick <b>Low</b>, <b>Medium</b>, or <b>Max</b> reasoning effort.
            </div>
        `;
        this.chatContainerEl.appendChild(bubble);
    }

    public clearConversation(): void {
        this.abortActiveStream();
        this.messages = [];
        if (this.chatContainerEl) {
            this.chatContainerEl.innerHTML = "";
            this.appendWelcomeBubble();
        }
        this.options.onStatusChange?.("Cleared AI Assistant conversation.");
    }

    public abortActiveStream(): void {
        if (this.activeAbortController) {
            this.activeAbortController.abort();
            this.activeAbortController = null;
        }
        this.setStreamingState(false);
    }

    private setStreamingState(streaming: boolean): void {
        this.isStreaming = streaming;
        if (this.inputEl) {
            this.inputEl.disabled = streaming;
        }
        if (this.sendBtnEl) {
            this.sendBtnEl.textContent = streaming ? "⏹" : "➡️";
            this.sendBtnEl.title = streaming ? "Stop AI Generation" : "Fast Send (Non-Reasoning Mode — Enter)";
        }
        this.updateBrainButtonLabel();
    }

    public formatMarkdown(md: string): string {
        let text = md.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
        text = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        text = text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="ha-ai-code-block"><code>$2</code></pre>');
        text = text.replace(/`([^`]+)`/g, '<code class="ha-ai-inline-code">$1</code>');
        text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
        text = text.replace(/\n/g, "<br/>");
        return text;
    }

    public async sendMessage(reasoningMode: AiReasoningMode = "none", explicitText?: string): Promise<void> {
        const text = (explicitText !== undefined ? explicitText : this.inputEl?.value || "").trim();
        if (!text || this.isStreaming) return;

        if (explicitText === undefined && this.inputEl) {
            this.inputEl.value = "";
        }
        this.closeBrainDropdown();
        this.lastUsedReasoningMode = reasoningMode;

        const userBubble = this.createElement("div", "ha-ai-bubble ha-ai-bubble-user");
        const modeTag =
            reasoningMode === "none"
                ? `<span class="ha-ai-mode-pill mode-fast">⚡ Fast</span>`
                : `<span class="ha-ai-mode-pill mode-reasoning">🧠 ${reasoningMode.toUpperCase()}</span>`;
        userBubble.innerHTML = `<div class="ha-ai-user-header">${modeTag}</div><div class="ha-ai-user-body"></div>`;
        (userBubble.querySelector(".ha-ai-user-body") as HTMLElement).textContent = text;
        this.chatContainerEl.appendChild(userBubble);
        this.scrollToBottom();

        this.messages.push({ role: "user", content: text });

        this.activeAbortController = new AbortController();
        this.setStreamingState(true);

        try {
            await this.runStreamTurn(reasoningMode, 0);
        } finally {
            this.activeAbortController = null;
            this.setStreamingState(false);
            if (this.inputEl) {
                this.inputEl.focus();
            }
        }
    }

    private async runStreamTurn(reasoningMode: AiReasoningMode, depth: number): Promise<void> {
        if (depth > 8) return;

        const thinkingBubble = this.createElement("div", "ha-ai-bubble ha-ai-thinking-bubble");
        thinkingBubble.innerHTML = `
            <div class="ha-ai-status-row">
                <span class="ha-ai-spinner"></span>
                <span class="status-label">${
                    reasoningMode === "none"
                        ? "Processing..."
                        : `Thinking (${reasoningMode.toUpperCase()} reasoning)...`
                }</span>
            </div>
            <div class="reasoning-container" style="display: none;"></div>
        `;
        this.chatContainerEl.appendChild(thinkingBubble);
        this.scrollToBottom();

        const statusLabelEl = thinkingBubble.querySelector(".status-label") as HTMLElement;
        const reasoningContainerEl = thinkingBubble.querySelector(".reasoning-container") as HTMLElement;

        let accumulatedText = "";
        let accumulatedReasoning = "";
        let aiTextBubble: HTMLElement | null = null;
        const toolCallsMap = new Map<number, StreamingToolCallEntry>();
        let hasExecutedTools = false;
        let streamError: string | null = null;

        const executePendingToolCalls = async () => {
            if (hasExecutedTools || toolCallsMap.size === 0) return;
            hasExecutedTools = true;

            if (!accumulatedReasoning.trim()) {
                thinkingBubble.remove();
            } else {
                const spinner = thinkingBubble.querySelector(".ha-ai-spinner") as HTMLElement | null;
                if (spinner) spinner.style.display = "none";
                statusLabelEl.textContent = `🧠 Thought (${reasoningMode.toUpperCase()})`;
                thinkingBubble.style.opacity = "0.85";
            }

            const orderedEntries = Array.from(toolCallsMap.values()).sort((a, b) => a.index - b.index);
            const assistantToolCallsPayload = orderedEntries.map(entry => ({
                id: entry.id || `call_${entry.name}_${Date.now()}`,
                type: "function" as const,
                function: {
                    name: entry.name,
                    arguments: entry.argumentsStr || "{}"
                }
            }));

            this.messages.push({
                role: "assistant",
                content: accumulatedText || null,
                tool_calls: assistantToolCallsPayload
            });

            for (const entry of orderedEntries) {
                const parsedArgs = parsePartialJson(entry.argumentsStr);
                const toolImpl = this.lookupAiTool(entry.name);
                let result: any = { status: "ok" };

                try {
                    if (toolImpl) {
                        result = await toolImpl.execute(this.state, parsedArgs);
                    } else {
                        result = { error: `Unknown tool '${entry.name}'` };
                    }
                    this.options.onAfterToolExecuted?.(entry.name, parsedArgs, result);

                    const doneLabel = this.toolRegistry.getCompletedToolStatusLabel(
                        entry.name,
                        parsedArgs,
                        result
                    );
                    entry.bubbleEl.classList.remove("streaming");
                    entry.bubbleEl.classList.add("completed");
                    entry.statusLabelEl.textContent = `✓ ${doneLabel}`;

                    if (result?.previewBase64) {
                        const imgWrap = this.createElement("div", "ha-ai-snapshot-preview");
                        const img = this.createElement("img", "ha-ai-snapshot-img", undefined, {
                            src: `data:image/png;base64,${result.previewBase64}`,
                            alt: "Canvas Snapshot"
                        });
                        imgWrap.appendChild(img);
                        entry.bubbleEl.appendChild(imgWrap);
                    }
                } catch (err: any) {
                    result = { error: err?.message || String(err) };
                    entry.bubbleEl.classList.remove("streaming");
                    entry.bubbleEl.classList.add("failed");
                    entry.statusLabelEl.textContent = `✕ ${entry.name} failed: ${result.error}`;
                }

                const llmResult =
                    result && typeof result === "object" && "previewBase64" in result
                        ? { ...result, previewBase64: "<omitted_base64_png_rendered_in_ui>" }
                        : result;

                this.messages.push({
                    role: "tool",
                    tool_call_id: entry.id || `call_${entry.name}`,
                    name: entry.name,
                    content: JSON.stringify(llmResult)
                });
                this.scrollToBottom();
            }

            await this.runStreamTurn(reasoningMode, depth + 1);
        };

        await this.aiClient.streamChatCompletion(
            this.messages,
            reasoningMode,
            (chunk) => {
                if (chunk.error) {
                    streamError = chunk.error;
                    return;
                }
                if (chunk.done) {
                    return;
                }

                const choice = chunk.choices?.[0];
                const delta = choice?.delta;

                if (delta) {
                    const reasoningDelta = delta.reasoning_content || delta.reasoning || delta.thought;
                    if (reasoningDelta) {
                        accumulatedReasoning += reasoningDelta;
                        statusLabelEl.textContent = `🧠 Reasoning (${reasoningMode.toUpperCase()})...`;
                        reasoningContainerEl.style.display = "block";
                        reasoningContainerEl.textContent = accumulatedReasoning;
                        this.scrollToBottom();
                    }

                    if (Array.isArray(delta.tool_calls)) {
                        for (const tc of delta.tool_calls) {
                            const idx = tc.index ?? 0;
                            let entry = toolCallsMap.get(idx);
                            if (!entry) {
                                const toolBubble = this.createElement(
                                    "div",
                                    "ha-ai-bubble ha-ai-tool-bubble streaming"
                                );
                                toolBubble.innerHTML = `
                                    <div class="ha-ai-tool-header">
                                        <span class="ha-ai-spinner"></span>
                                        <span class="tool-status-label">🔧 Preparing tool...</span>
                                    </div>
                                    <div class="tool-args-live-preview"></div>
                                `;
                                this.chatContainerEl.appendChild(toolBubble);
                                entry = {
                                    index: idx,
                                    id: tc.id || `call_${idx}_${Date.now()}`,
                                    name: "",
                                    argumentsStr: "",
                                    bubbleEl: toolBubble,
                                    statusLabelEl: toolBubble.querySelector(".tool-status-label") as HTMLElement,
                                    argsPreviewEl: toolBubble.querySelector(".tool-args-live-preview") as HTMLElement
                                };
                                toolCallsMap.set(idx, entry);
                            }

                            if (tc.id) entry.id = tc.id;
                            if (tc.function?.name) {
                                if (!entry.name) {
                                    entry.name = tc.function.name;
                                } else if (tc.function.name !== entry.name) {
                                    entry.name += tc.function.name;
                                }
                            }
                            if (tc.function?.arguments) entry.argumentsStr += tc.function.arguments;

                            const partialArgs = parsePartialJson(entry.argumentsStr);
                            entry.statusLabelEl.textContent = this.toolRegistry.getLiveToolStatusLabel(
                                entry.name,
                                partialArgs,
                                entry.argumentsStr
                            );
                            this.renderLiveToolArgsPills(entry.argsPreviewEl, partialArgs);
                            this.scrollToBottom();
                        }
                    }

                    if (delta.content) {
                        accumulatedText += delta.content;

                        const thinkMatch = accumulatedText.match(/<think>([\s\S]*?)(?:<\/think>|$)/i);
                        if (thinkMatch) {
                            statusLabelEl.textContent = `🧠 Reasoning (${reasoningMode.toUpperCase()})...`;
                            reasoningContainerEl.style.display = "block";
                            reasoningContainerEl.textContent = thinkMatch[1].trim();
                        }

                        const cleanVisible = accumulatedText
                            .replace(/<think>[\s\S]*?<\/think>/gi, "")
                            .replace(/<think>[\s\S]*$/i, "")
                            .trim();

                        if (cleanVisible.length > 0) {
                            if (!accumulatedReasoning.trim() && !thinkMatch) {
                                thinkingBubble.remove();
                            } else {
                                const spinner = thinkingBubble.querySelector(".ha-ai-spinner") as HTMLElement | null;
                                if (spinner) spinner.style.display = "none";
                                statusLabelEl.textContent = `🧠 Thought (${
                                    reasoningMode === "none" ? "AUTO" : reasoningMode.toUpperCase()
                                })`;
                            }
                            if (!aiTextBubble) {
                                aiTextBubble = this.createElement(
                                    "div",
                                    "ha-ai-bubble ha-ai-bubble-assistant"
                                );
                                this.chatContainerEl.appendChild(aiTextBubble);
                            }
                            aiTextBubble.innerHTML = this.formatMarkdown(cleanVisible);
                            this.scrollToBottom();
                        }
                    }
                }
            },
            this.activeAbortController?.signal
        );

        if (streamError) {
            thinkingBubble.remove();
            const errBubble = this.createElement("div", "ha-ai-bubble ha-ai-bubble-error");
            errBubble.innerHTML = `
                <div class="ha-ai-error-title">✕ ${this.aiClient.getProviderDisplayName()} Connection Notice</div>
                <div class="ha-ai-error-body">${streamError}</div>
            `;
            this.chatContainerEl.appendChild(errBubble);
            this.scrollToBottom();
            return;
        }

        if (toolCallsMap.size > 0 && !hasExecutedTools) {
            await executePendingToolCalls();
            return;
        }

        if (accumulatedText.trim()) {
            this.messages.push({ role: "assistant", content: accumulatedText });
        } else if (!accumulatedReasoning.trim()) {
            thinkingBubble.remove();
        }
    }

    private renderLiveToolArgsPills(container: HTMLElement, partialArgs: Record<string, any>): void {
        const entries = Object.entries(partialArgs).filter(([, v]) => v !== undefined && v !== null);
        if (entries.length === 0) {
            container.innerHTML = "";
            return;
        }
        container.innerHTML = entries
            .slice(0, 6)
            .map(([k, v]) => {
                const valStr =
                    typeof v === "object" ? JSON.stringify(v).slice(0, 36) : String(v).slice(0, 36);
                return `<span class="ha-ai-arg-pill"><b>${k}:</b> ${valStr}</span>`;
            })
            .join("");
    }

    public scrollToBottom(): void {
        if (this.chatContainerEl) {
            this.chatContainerEl.scrollTop = this.chatContainerEl.scrollHeight;
        }
    }

    /**
     * Returns the list of all built-in + custom registered AI providers for multi-provider UI dropdowns.
     */
    public static getProviderSelectOptions(): Array<{ label: string; value: ProviderType | string }> {
        const builtIn: Array<{ label: string; value: ProviderType | string }> = [
            { label: "🏠 LM Studio (Local Server — default http://localhost:1234)", value: "lmstudio" },
            { label: "🟢 OpenAI (GPT-6, GPT-4o, o3-mini, o1)", value: "openai" },
            { label: "🟠 Anthropic Claude (Claude Opus 4.8, Sonnet 5, 3.7 Sonnet)", value: "claude" },
            { label: "🔵 Google Gemini (API Key — Gemini 3.8 / 2.5 Pro & Flash)", value: "gemini" },
            { label: "🌐 Google Gemini (Google One OAuth Subscription)", value: "gemini-googleone" },
            { label: "⚫ xAI Grok (Grok 4.7, Grok-2)", value: "grok" },
            { label: "🔌 Custom OpenAI-Compatible Endpoint (Ollama / vLLM / OpenRouter)", value: "custom-openai" },
            { label: "🔌 Custom Anthropic Claude Endpoint", value: "custom-claude" }
        ];
        for (const custom of AiProviderRegistry.getAllProviders()) {
            builtIn.push({
                label: `🧩 ${custom.displayName} (${custom.providerId})`,
                value: custom.providerId
            });
        }
        return builtIn;
    }
}
