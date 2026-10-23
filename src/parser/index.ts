import { EventEmitter } from "node:events";

export interface ToolCallStart {
    id: string;
    name: string;
}

export interface ToolCallChunk {
    id: string;
    chunk: string;
}

export interface ToolCallEnd {
    id: string;
    name: string;
    args: string;
}

/**
 * Tolerant partial JSON parser (adapted from `episode-caster`'s `AiChatDialog.parsePartialJson`)
 * that auto-closes unclosed quotes, arrays, and objects so incomplete streaming tool call arguments
 * can be live-rendered into human-readable tool labels as tokens stream in from any LLM provider.
 */
export function parsePartialJson(jsonStr: string): Record<string, any> {
    if (!jsonStr || !jsonStr.trim()) return {};
    try {
        const parsed = JSON.parse(jsonStr);
        return typeof parsed === "object" && parsed !== null ? parsed : {};
    } catch {
        let s = jsonStr.trim();
        let inString = false;
        let escape = false;
        const stack: string[] = [];

        for (let i = 0; i < s.length; i++) {
            const c = s[i];
            if (escape) {
                escape = false;
                continue;
            }
            if (c === "\\") {
                escape = true;
                continue;
            }
            if (c === '"') {
                inString = !inString;
                continue;
            }
            if (!inString) {
                if (c === "{") stack.push("}");
                else if (c === "[") stack.push("]");
                else if (c === "}" || c === "]") {
                    if (stack.length > 0 && stack[stack.length - 1] === c) {
                        stack.pop();
                    }
                }
            }
        }

        if (inString) s += '"';
        s = s.replace(/,\s*$/, "");
        if (s.match(/:\s*$/)) s += "null";
        while (stack.length > 0) {
            s += stack.pop();
        }

        try {
            const recovered = JSON.parse(s);
            return typeof recovered === "object" && recovered !== null ? recovered : {};
        } catch {
            return {};
        }
    }
}

/**
 * Parses a continuous text stream from an LLM, detecting tool calls in the format `tool_name(...)`.
 * Emits events for text, tool starts, argument chunks, and tool completion.
 */
export class ToolParser extends EventEmitter {
    private buffer = "";
    private state: "TEXT" | "TOOL_ARGS" = "TEXT";
    private currentToolName = "";
    private currentToolArgs = "";
    private currentToolId = "";

    private depth = 0;
    private inString = false;
    private escapeNext = false;
    private stringChar = "";

    private registeredTools: Set<string>;

    constructor(registeredTools: string[]) {
        super();
        this.registeredTools = new Set(registeredTools);
    }

    public setRegisteredTools(registeredTools: string[]): void {
        this.registeredTools = new Set(registeredTools);
    }

    /**
     * Feeds a chunk of text into the parser.
     */
    public feed(chunk: string): void {
        for (const char of chunk) {
            if (this.state === "TEXT") {
                this.buffer += char;

                for (const tool of this.registeredTools) {
                    if (this.buffer.endsWith(tool + "(")) {
                        const textBefore = this.buffer.slice(0, -(tool.length + 1));
                        if (textBefore) {
                            this.emit("text", textBefore);
                        }
                        this.buffer = "";
                        this.state = "TOOL_ARGS";
                        this.currentToolName = tool;
                        this.currentToolArgs = "";
                        this.currentToolId = "tool_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
                        this.depth = 1;
                        this.inString = false;
                        this.escapeNext = false;

                        this.emit("toolStart", {
                            id: this.currentToolId,
                            name: this.currentToolName
                        } as ToolCallStart);
                        break;
                    }
                }
            } else if (this.state === "TOOL_ARGS") {
                if (this.escapeNext) {
                    this.escapeNext = false;
                    this.currentToolArgs += char;
                    this.emit("toolArgsChunk", {
                        id: this.currentToolId,
                        chunk: char
                    } as ToolCallChunk);
                    continue;
                }

                if (char === "\\") {
                    this.escapeNext = true;
                } else if (this.inString) {
                    if (char === this.stringChar) {
                        this.inString = false;
                    }
                } else {
                    if (char === '"' || char === "'" || char === "`") {
                        this.inString = true;
                        this.stringChar = char;
                    } else if (char === "(" || char === "{" || char === "[") {
                        this.depth++;
                    } else if (char === ")" || char === "}" || char === "]") {
                        this.depth--;
                    }
                }

                if (this.depth === 0 && char === ")") {
                    this.emit("toolEnd", {
                        id: this.currentToolId,
                        name: this.currentToolName,
                        args: this.currentToolArgs
                    } as ToolCallEnd);
                    this.state = "TEXT";
                    this.currentToolName = "";
                    this.currentToolArgs = "";
                    this.currentToolId = "";
                } else {
                    this.currentToolArgs += char;
                    this.emit("toolArgsChunk", {
                        id: this.currentToolId,
                        chunk: char
                    } as ToolCallChunk);
                }
            }
        }
    }

    /**
     * Signals the end of the stream. Flushes any remaining text.
     */
    public end(): void {
        if (this.state === "TEXT" && this.buffer) {
            this.emit("text", this.buffer);
            this.buffer = "";
        } else if (this.state === "TOOL_ARGS") {
            this.emit("text", this.currentToolName + "(" + this.currentToolArgs);
        }
    }
}

/**
 * Extracts `<think>...</think>` blocks from accumulated streaming assistant text so UI components
 * can live-render collapsible reasoning blocks alongside visible reply text.
 */
export function extractLiveThinkingAndReply(
    rawAccumulatedContent: string,
    explicitReasoningAccumulated: string = ""
): { thinkingText: string; visibleReplyText: string; isThinkingOpen: boolean } {
    let thinkingText = explicitReasoningAccumulated;
    let visibleReplyText = rawAccumulatedContent;
    let isThinkingOpen = false;

    const thinkStartIdx = visibleReplyText.indexOf("<think>");
    if (thinkStartIdx !== -1) {
        const thinkEndIdx = visibleReplyText.indexOf("</think>");
        if (thinkEndIdx !== -1 && thinkEndIdx > thinkStartIdx) {
            const extractedThink = visibleReplyText.slice(thinkStartIdx + 7, thinkEndIdx);
            thinkingText = (thinkingText ? thinkingText + "\n" : "") + extractedThink;
            visibleReplyText =
                visibleReplyText.slice(0, thinkStartIdx) +
                visibleReplyText.slice(thinkEndIdx + 8);
        } else {
            isThinkingOpen = true;
            const partialThink = visibleReplyText.slice(thinkStartIdx + 7);
            thinkingText = (thinkingText ? thinkingText + "\n" : "") + partialThink;
            visibleReplyText = visibleReplyText.slice(0, thinkStartIdx);
        }
    }

    return {
        thinkingText: thinkingText.trim(),
        visibleReplyText: visibleReplyText.trim(),
        isThinkingOpen
    };
}
