/**
 * Monitoring System - Track template execution and trigger rollbacks
 * Simplified version without Claude Code SDK integration
 */
import { OptimizationTemplate, MonitoringMetric, RollbackTrigger } from '../types/template.js';
export declare class MonitoringSystem {
    private activeMonitors;
    private metricsHistory;
    /**
     * Start monitoring a template execution
     */
    startMonitoring(template: OptimizationTemplate, executionId: string): Promise<void>;
    /**
     * Stop monitoring a template execution
     */
    stopMonitoring(executionId: string): void;
    /**
     * Main monitoring loop
     */
    private monitoringLoop;
    /**
     * Collect metrics using simplified commands
     */
    private collectMetrics;
    /**
     * Check if any rollback triggers are activated
     */
    private checkRollbackTriggers;
    /**
     * Evaluate if a trigger condition is met
     */
    private evaluateTriggerCondition;
    /**
     * Extract threshold value from condition string
     */
    private extractThreshold;
    /**
     * Handle activated rollback triggers
     */
    private handleRollbackTriggers;
    /**
     * Execute automatic rollback
     */
    private executeAutomaticRollback;
    /**
     * Send alert for investigation
     */
    private sendAlert;
    /**
     * Escalate issue
     */
    private escalateIssue;
    /**
     * Record metrics for historical analysis
     */
    private recordMetrics;
    /**
     * Get current metrics for an execution
     */
    private getCurrentMetrics;
    /**
     * Get metrics history for analysis
     */
    getMetricsHistory(executionId: string): MetricSnapshot[];
    /**
     * Get active monitoring sessions
     */
    getActiveMonitors(): MonitoringSession[];
}
interface MonitoringSession {
    template: OptimizationTemplate;
    executionId: string;
    startTime: Date;
    metrics: MonitoringMetric[];
    triggers: RollbackTrigger[];
    isActive: boolean;
}
interface MetricSnapshot {
    timestamp: Date;
    metrics: Record<string, number>;
}
export {};
