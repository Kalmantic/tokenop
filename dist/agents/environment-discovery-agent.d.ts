/**
 * Environment Discovery Agent - Discovers the full LLM stack
 * Analyzes Application, Serving, and Infrastructure layers for optimization opportunities
 */
import { EnvironmentProfile } from '../types/template.js';
export declare class EnvironmentDiscoveryAgent {
    /**
     * Stage 2: Environment Discovery ⏳
     * Discover the full LLM stack
     */
    discoverEnvironment(): Promise<EnvironmentProfile>;
    /**
     * Discover Application Layer
     */
    private discoverApplicationLayer;
    /**
     * Discover Serving Layer
     */
    private discoverServingLayer;
    /**
     * Discover Infrastructure Layer
     */
    private discoverInfrastructureLayer;
    /**
     * Try to detect GPUs using nvidia-smi
     */
    private detectGPUs;
    /**
     * Search for patterns in files
     */
    private searchInFiles;
    /**
     * Find files matching pattern
     */
    private findFiles;
    /**
     * Get GPU bandwidth by name
     */
    private getGPUBandwidth;
    /**
     * Get GPU hourly cost by name
     */
    private getGPUCost;
}
