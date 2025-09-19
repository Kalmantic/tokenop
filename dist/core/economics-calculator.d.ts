/**
 * Economics Calculator - Implements TokenSqueeze economic principles in template formulas
 * Calculates baseline costs, projected savings, and ROI using template economics models
 */
import { OptimizationTemplate, EnvironmentProfile, EconomicsModel } from '../types/template.js';
export declare class EconomicsCalculator {
    private environmentVars;
    /**
     * Calculate template impact on economics
     */
    calculateTemplateImpact(template: OptimizationTemplate, environment: EnvironmentProfile): Promise<any>;
    /**
     * Calculate baseline economics from template and environment
     */
    calculateBaseline(template: OptimizationTemplate, environment: EnvironmentProfile): Promise<Record<string, number>>;
    /**
     * Calculate projected improvements from template
     */
    calculateProjectedSavings(template: OptimizationTemplate, baseline: Record<string, number>): Promise<Record<string, number>>;
    /**
     * Calculate ROI from baseline, optimized metrics, and economics model
     */
    calculateROI(baseline: Record<string, number>, optimized: Record<string, number>, economics: EconomicsModel): number;
    /**
     * Extract environment variables for formula evaluation
     */
    private extractEnvironmentVariables;
    /**
     * Evaluate formula string with environment variables
     */
    private evaluateFormula;
    /**
     * Calculate memory bandwidth utilization (TokenSqueeze principle)
     */
    private calculateMemoryBandwidthUtilization;
    /**
     * Calculate arithmetic intensity (TokenSqueeze core metric)
     */
    private calculateArithmeticIntensity;
    /**
     * Calculate context length tax (TokenSqueeze KV cache principle)
     */
    private calculateContextLengthTax;
    /**
     * Calculate batch efficiency (TokenSqueeze batching principle)
     */
    private calculateBatchEfficiency;
    /**
     * Format currency for display
     */
    private formatCurrency;
    /**
     * Generate economics report
     */
    generateEconomicsReport(template: OptimizationTemplate, baseline: Record<string, number>, projected: Record<string, number>): string;
}
