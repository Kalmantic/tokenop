/**
 * Template Execution Agent - Executes optimization templates
 * Simplified version without Claude Code SDK integration
 */
import { OptimizationTemplate, EnvironmentProfile, TemplateExecutionResult } from '../types/template.js';
export declare class TemplateExecutionAgent {
    private economicsCalculator;
    private executionHistory;
    constructor();
    /**
     * Stage 3: Template Execution ⏳
     * Execute a template with simplified logic
     */
    executeTemplate(template: OptimizationTemplate, environment: EnvironmentProfile, options?: ExecutionOptions): Promise<TemplateExecutionResult>;
    /**
     * Validate prerequisites
     */
    private validatePrerequisites;
    /**
     * Execute optimization steps
     */
    private executeOptimizationSteps;
    /**
     * Execute a single step
     */
    private executeStep;
    /**
     * Measure optimization impact
     */
    private measureOptimizationImpact;
    /**
     * Rollback execution if something goes wrong
     */
    private rollbackExecution;
    /**
     * Get execution history
     */
    getExecutionHistory(): TemplateExecutionResult[];
    /**
     * Get specific execution result
     */
    getExecutionResult(executionId: string): TemplateExecutionResult | undefined;
}
export interface ExecutionOptions {
    dryRun?: boolean;
    skipPrerequisites?: boolean;
    customEnvironmentVars?: Record<string, string>;
    rollbackOnFailure?: boolean;
}
