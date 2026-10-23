export type AiToolCategory =
    | "inspect"
    | "canvas"
    | "layer"
    | "vector"
    | "raster"
    | "filter"
    | "shared_object"
    | "library"
    | "animation"
    | "file"
    | "code"
    | "agent"
    | "system"
    | string;

export interface OpenAiToolSchema {
    type: "function";
    function: {
        name: string;
        description: string;
        parameters: {
            type: "object";
            properties: Record<string, any>;
            required?: string[];
        };
    };
}

/**
 * Context-specific availability rule for a Tool or AbstractAiTool.
 * Enables tools that are only available in specific states—such as AICore's tools that unlock
 * only after an implementation plan or task list is created (`requiresPlanOrTaskList`), tools
 * that are only available during initial discovery (`hideAfterPlanOrTaskList`), or tools scoped
 * to a specific editing context (`when`).
 */
export interface ToolContextRule<TContext = any> {
    /**
     * Custom predicate evaluated against the current agent/editor context.
     */
    when?: (context: TContext) => boolean;
    /**
     * If true, the tool is only available after `context.hasTaskListBeenCreated` or
     * `context.hasImplementationPlan` is truthy.
     */
    requiresPlanOrTaskList?: boolean;
    /**
     * If true, the tool is hidden once `context.hasTaskListBeenCreated` or
     * `context.hasImplementationPlan` is truthy (e.g. `@create_task_list`).
     */
    hideAfterPlanOrTaskList?: boolean;
    /**
     * Optional category tag (e.g. "inspect", "vector", "raster", "code").
     */
    category?: AiToolCategory;
    /**
     * Optional icon for live UI rendering.
     */
    icon?: string;
}

/**
 * Represents a tool available to an AI agent or UI chat session.
 * Handles JSON argument parsing, validation, execution, OpenAI function schema export,
 * and context-dependent availability predicates.
 */
export class Tool<TContext = any> {
    public readonly name: string;
    public readonly description: string;
    public readonly jsonSchema: Record<string, any>;
    public readonly contextRule?: ToolContextRule<TContext>;
    private readonly executor: (args: any, context?: TContext) => Promise<string>;

    constructor(
        name: string,
        description: string,
        jsonSchema: Record<string, any>,
        executor: (args: any, context?: TContext) => Promise<string>,
        contextRule?: ToolContextRule<TContext>
    ) {
        this.name = name;
        this.description = description;
        this.jsonSchema = jsonSchema || { properties: {} };
        this.executor = executor;
        this.contextRule = contextRule;
    }

    /**
     * Evaluates whether this tool is available in the given execution context.
     * Supports both explicit `ToolContextRule` predicates and AICore's `@`-prefix convention
     * when `context` provides `hasTaskListBeenCreated` or `hasImplementationPlan`.
     */
    public isAvailableInContext(context?: TContext): boolean {
        if (!context) return true;

        if (this.contextRule?.when) {
            return this.contextRule.when(context);
        }

        const anyCtx = context as any;
        const planReady =
            Boolean(anyCtx.hasTaskListBeenCreated) ||
            Boolean(anyCtx.hasImplementationPlan);

        if (this.contextRule?.hideAfterPlanOrTaskList && planReady) {
            return false;
        }
        if (this.contextRule?.requiresPlanOrTaskList && !planReady) {
            return false;
        }

        // AICore convention when `enforcePlanGate` is active on the context:
        // Before a task list / plan is created, only `@`-prefixed discovery tools are available;
        // once created, `@create_task_list` is hidden and all other tools unlock.
        if (anyCtx.enforcePlanGate) {
            if (planReady) {
                return this.name !== "@create_task_list";
            }
            return this.name.startsWith("@");
        }

        return true;
    }

    /**
     * Parses a raw JSON argument string, validates it, and invokes the tool's underlying async function.
     */
    public async callTool(jsonString: string, context?: TContext): Promise<string> {
        let parsedArgs: any;
        try {
            parsedArgs = jsonString && jsonString.trim() ? JSON.parse(jsonString) : {};
        } catch (err: any) {
            throw new Error(`Invalid JSON arguments provided for tool ${this.name}: ${err.message}`);
        }

        this.validateArgs(parsedArgs);
        return await this.executor(parsedArgs, context);
    }

    /**
     * Validates parsed arguments against the expected schema/types.
     */
    private validateArgs(args: any): void {
        if (this.jsonSchema.required && Array.isArray(this.jsonSchema.required)) {
            for (const reqProp of this.jsonSchema.required) {
                if (args[reqProp] === undefined) {
                    throw new Error(`Missing required parameter '${reqProp}' for tool '${this.name}'`);
                }
            }
        }
    }

    /**
     * Returns the formatted prompt representation of this tool for injection into text-based system prompts.
     */
    public toPromptString(): string {
        return `// ${this.description}\n${this.name}(${JSON.stringify(this.jsonSchema.properties || {})})`;
    }

    /**
     * Returns the OpenAI/Anthropic/Gemini-compatible function tool schema.
     */
    public toOpenAiSchema(): OpenAiToolSchema {
        return {
            type: "function",
            function: {
                name: this.name.replace(/^@/, ""),
                description: this.description,
                parameters: {
                    type: "object",
                    properties: this.jsonSchema.properties || {},
                    ...(Array.isArray(this.jsonSchema.required) && this.jsonSchema.required.length > 0
                        ? { required: this.jsonSchema.required }
                        : {})
                }
            }
        };
    }
}

/**
 * Abstract parent class for structured AI tools with live streaming status labels,
 * OpenAI function schema generation, and context-aware availability.
 */
export abstract class AbstractAiTool<TContext = any> {
    public abstract readonly name: string;
    public abstract readonly category: AiToolCategory;
    public abstract readonly icon: string;
    public abstract readonly description: string;
    public abstract readonly parameters: {
        type: "object";
        properties: Record<string, any>;
        required?: string[];
    };

    /**
     * Optional override to make this tool available only in specific contexts
     * (e.g., after an implementation plan is created, or when a specific layer/mode is active).
     */
    public isAvailableInContext(_context?: TContext): boolean {
        return true;
    }

    public toOpenAiSchema(): OpenAiToolSchema {
        return {
            type: "function",
            function: {
                name: this.name,
                description: this.description,
                parameters: this.parameters
            }
        };
    }

    /**
     * Converts this `AbstractAiTool` into a `Tool` instance usable by `AICore` agents and `ToolRegistry`.
     */
    public toTool(
        contextGetter: () => TContext,
        options?: { nameOverride?: string; contextRule?: ToolContextRule<TContext> }
    ): Tool<TContext> {
        const toolName = options?.nameOverride || this.name;
        return new Tool<TContext>(
            toolName,
            this.description,
            this.parameters,
            async (args, explicitCtx) => {
                const ctx = explicitCtx ?? contextGetter();
                const res = await this.execute(ctx, args);
                return typeof res === "string" ? res : JSON.stringify(res, null, 2);
            },
            options?.contextRule || {
                category: this.category,
                icon: this.icon,
                when: (ctx) => this.isAvailableInContext(ctx)
            }
        );
    }

    /**
     * Returns a live human-readable status label while arguments are streaming in token-by-token.
     */
    public abstract getLiveStatusLabel(partialArgs: Record<string, any>, rawArgsStr: string): string;

    /**
     * Returns a completed human-readable status label after the tool has executed.
     */
    public abstract getCompletedStatusLabel(args: Record<string, any>, result: any): string;

    /**
     * Executes the tool against the given `TContext`.
     */
    public abstract execute(context: TContext, args: Record<string, any>): Promise<any> | any;
}

/**
 * Context-aware tool registry supporting both `Tool` and `AbstractAiTool<TContext>`.
 * Allows projects like `AICore`, `HACanvas`, and `HAEditor` to register global tools alongside
 * context-specific tools (such as tools that unlock only after an implementation plan / task list
 * is created, or tools scoped to specific canvas/layer states).
 */
export class ContextualToolRegistry<TContext = any> {
    private readonly toolsMap: Map<string, Tool<TContext>> = new Map();
    private readonly aiToolsMap: Map<string, AbstractAiTool<TContext>> = new Map();
    private readonly rulesMap: Map<string, ToolContextRule<TContext>> = new Map();

    constructor(initialAiTools: AbstractAiTool<TContext>[] = []) {
        for (const t of initialAiTools) {
            this.registerAiTool(t);
        }
    }

    public registerTool(tool: Tool<TContext>, rule?: ToolContextRule<TContext>): void {
        this.toolsMap.set(tool.name, tool);
        if (rule || tool.contextRule) {
            this.rulesMap.set(tool.name, { ...tool.contextRule, ...rule });
        }
    }

    public registerAiTool(
        aiTool: AbstractAiTool<TContext>,
        rule?: ToolContextRule<TContext>,
        contextGetter?: () => TContext
    ): void {
        this.aiToolsMap.set(aiTool.name, aiTool);
        if (rule) {
            this.rulesMap.set(aiTool.name, rule);
        }
        const wrapped = aiTool.toTool(
            contextGetter || (() => ({} as TContext)),
            { contextRule: rule }
        );
        this.toolsMap.set(aiTool.name, wrapped);
    }

    public unregisterTool(name: string): void {
        this.toolsMap.delete(name);
        this.aiToolsMap.delete(name);
        this.rulesMap.delete(name);
    }

    public getTool(name: string): Tool<TContext> | undefined {
        return this.toolsMap.get(name);
    }

    public getAiTool(name: string): AbstractAiTool<TContext> | undefined {
        return this.aiToolsMap.get(name);
    }

    public getAllTools(): Tool<TContext>[] {
        return Array.from(this.toolsMap.values());
    }

    public getAllAiTools(): AbstractAiTool<TContext>[] {
        return Array.from(this.aiToolsMap.values());
    }

    /**
     * Returns only the tools that are active for the provided `context`.
     */
    public getActiveTools(context?: TContext): Tool<TContext>[] {
        if (!context) {
            return this.getAllTools();
        }
        return this.getAllTools().filter((tool) => {
            const rule = this.rulesMap.get(tool.name);
            if (rule?.when && !rule.when(context)) {
                return false;
            }
            const aiTool = this.aiToolsMap.get(tool.name);
            if (aiTool && !aiTool.isAvailableInContext(context)) {
                return false;
            }
            return tool.isAvailableInContext(context);
        });
    }

    /**
     * Returns only the `AbstractAiTool` instances that are active for the provided `context`.
     */
    public getActiveAiTools(context?: TContext): AbstractAiTool<TContext>[] {
        if (!context) {
            return this.getAllAiTools();
        }
        return this.getAllAiTools().filter((aiTool) => {
            const rule = this.rulesMap.get(aiTool.name);
            if (rule?.when && !rule.when(context)) {
                return false;
            }
            return aiTool.isAvailableInContext(context);
        });
    }

    /**
     * Returns the names of all tools currently active in the given `context`.
     */
    public getActiveToolNames(context?: TContext): string[] {
        return this.getActiveTools(context).map((t) => t.name);
    }

    /**
     * Generates the OpenAI function-calling schema list for all active tools in `context`.
     */
    public getOpenAiToolsSchema(context?: TContext): OpenAiToolSchema[] {
        if (this.aiToolsMap.size > 0) {
            return this.getActiveAiTools(context).map((t) => t.toOpenAiSchema());
        }
        return this.getActiveTools(context).map((t) => t.toOpenAiSchema());
    }

    /**
     * Builds the `## Available Tools` prompt section for text-stream tool parsers (like AICore).
     */
    public buildToolsPromptSection(context?: TContext): string {
        let out = "\n\n## Available Tools\n";
        for (const tool of this.getActiveTools(context)) {
            out += tool.toPromptString() + "\n";
        }
        return out;
    }

    /**
     * Executes a registered `AbstractAiTool` or `Tool` against `context`.
     */
    public async executeTool(
        name: string,
        context: TContext,
        args: Record<string, any> = {}
    ): Promise<any> {
        const aiTool = this.aiToolsMap.get(name);
        if (aiTool) {
            if (!aiTool.isAvailableInContext(context)) {
                throw new Error(`Tool '${name}' is not available in the current context.`);
            }
            return await aiTool.execute(context, args);
        }
        const stdTool = this.toolsMap.get(name);
        if (!stdTool) {
            throw new Error(`Tool '${name}' is not registered.`);
        }
        if (!stdTool.isAvailableInContext(context)) {
            throw new Error(`Tool '${name}' is not available in the current context.`);
        }
        return await stdTool.callTool(JSON.stringify(args), context);
    }

    public getLiveToolStatusLabel(
        toolName: string,
        partialArgs: Record<string, any>,
        rawArgsStr: string
    ): string {
        const aiTool = this.aiToolsMap.get(toolName);
        if (aiTool) {
            return `${aiTool.icon} ${aiTool.getLiveStatusLabel(partialArgs, rawArgsStr)}`;
        }
        const rule = this.rulesMap.get(toolName);
        const icon = rule?.icon || "🔧";
        return `${icon} Running ${toolName}...`;
    }

    public getCompletedToolStatusLabel(
        toolName: string,
        args: Record<string, any>,
        result: any
    ): string {
        const aiTool = this.aiToolsMap.get(toolName);
        if (aiTool) {
            return aiTool.getCompletedStatusLabel(args, result);
        }
        return `Executed ${toolName}`;
    }
}
