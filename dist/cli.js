#!/usr/bin/env node
"use strict";
/**
 * TokenOp CLI - Template-driven LLM optimization using Claude Code SDK
 * Main entry point for the SLC (Simple, Lovable, Complete) implementation
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const ora_1 = __importDefault(require("ora"));
const template_engine_js_1 = require("./core/template-engine.js");
const template_execution_agent_js_1 = require("./agents/template-execution-agent.js");
const environment_discovery_agent_js_1 = require("./agents/environment-discovery-agent.js");
const economics_calculator_js_1 = require("./core/economics-calculator.js");
const program = new commander_1.Command();
// Global instances
const templateEngine = new template_engine_js_1.TemplateEngine();
const executionAgent = new template_execution_agent_js_1.TemplateExecutionAgent();
const discoveryAgent = new environment_discovery_agent_js_1.EnvironmentDiscoveryAgent();
const economicsCalculator = new economics_calculator_js_1.EconomicsCalculator();
/**
 * Main discover command - The One Perfect Command
 */
program
    .command('discover')
    .description('🔍 Discover optimization opportunities across your full LLM stack')
    .option('--dry-run', 'Simulate optimizations without making changes')
    .option('--templates-dir <dir>', 'Custom templates directory')
    .option('--output <file>', 'Save results to file')
    .action(async (options) => {
    console.log(chalk_1.default.blue.bold('\n🚀 TokenOp: Full-Stack LLM Optimization Discovery\n'));
    console.log(chalk_1.default.gray('Using template-driven Claude Code SDK agents\n'));
    const spinner = (0, ora_1.default)('Starting discovery...').start();
    try {
        // Stage 1: Load Templates ⏳
        spinner.text = 'Loading optimization templates...';
        await templateEngine.loadTemplates();
        const allTemplates = templateEngine.listTemplates();
        spinner.succeed(`Loaded ${allTemplates.length} optimization templates`);
        // Stage 2: Environment Discovery ⏳
        spinner.start('Discovering your LLM infrastructure...');
        const environment = await discoveryAgent.discoverEnvironment();
        spinner.succeed('Environment discovery complete');
        console.log(chalk_1.default.cyan('\n📊 Environment Summary:'));
        console.log(`  Application: ${environment.application.runtime_detected.join(', ') || 'None detected'}`);
        console.log(`  Serving: ${environment.serving.frameworks_detected.join(', ') || 'None detected'}`);
        console.log(`  Infrastructure: ${environment.infrastructure.gpu_inventory.length} GPU(s) detected`);
        // Stage 3: Template Matching ⏳
        spinner.start('Finding applicable optimization templates...');
        const matchingTemplates = await templateEngine.findMatchingTemplates(environment);
        spinner.succeed(`Found ${matchingTemplates.length} applicable templates`);
        if (matchingTemplates.length === 0) {
            console.log(chalk_1.default.yellow('\n⚠️  No optimization templates match your current environment.'));
            console.log(chalk_1.default.gray('This might indicate:\n'));
            console.log(chalk_1.default.gray('  • Your infrastructure is already optimized'));
            console.log(chalk_1.default.gray('  • You need custom templates for your setup'));
            console.log(chalk_1.default.gray('  • Environment detection needs refinement\n'));
            return;
        }
        // Stage 4: Economic Analysis ⏳
        spinner.start('Calculating optimization economics...');
        const optimizationPlans = await analyzeOptimizationEconomics(matchingTemplates, environment);
        spinner.succeed('Economic analysis complete');
        // Stage 5: Present Results 📊
        console.log(chalk_1.default.green.bold('\n💡 Optimization Opportunities Found:\n'));
        let totalSavings = 0;
        let totalImplementationCost = 0;
        for (let i = 0; i < Math.min(5, optimizationPlans.length); i++) {
            const plan = optimizationPlans[i];
            console.log(chalk_1.default.blue(`${i + 1}. ${plan.template.name}`));
            console.log(`   ${chalk_1.default.gray(plan.template.description)}`);
            console.log(`   💰 Monthly Savings: ${chalk_1.default.green('$' + plan.projectedSavings.toLocaleString())}`);
            console.log(`   🔧 Implementation: ${plan.template.optimization.effort_estimate} (${plan.template.optimization.risk_level} risk)`);
            console.log(`   📈 ROI: ${chalk_1.default.cyan(plan.roi.toFixed(1) + '%')} annually`);
            console.log(`   🎯 Confidence: ${chalk_1.default.yellow((plan.template.confidence * 100).toFixed(1) + '%')}`);
            console.log('');
            totalSavings += plan.projectedSavings;
            totalImplementationCost += plan.template.economics.implementation_cost.total_cost;
        }
        // Summary
        console.log(chalk_1.default.green.bold('📈 Total Economic Impact:'));
        console.log(`   💰 Total Monthly Savings: ${chalk_1.default.green.bold('$' + totalSavings.toLocaleString())}`);
        console.log(`   💸 Total Implementation Cost: ${chalk_1.default.red('$' + totalImplementationCost.toLocaleString())}`);
        console.log(`   📊 Combined ROI: ${chalk_1.default.cyan.bold(((totalSavings * 12 - totalImplementationCost) / totalImplementationCost * 100).toFixed(1) + '%')}`);
        console.log(`   ⏱️  Payback Period: ${chalk_1.default.blue((totalImplementationCost / totalSavings).toFixed(1) + ' months')}\n`);
        // Next Steps
        console.log(chalk_1.default.blue.bold('🚀 Next Steps:'));
        if (options.dryRun) {
            console.log(`   ${chalk_1.default.gray('└')} Run without --dry-run to execute optimizations`);
        }
        else {
            console.log(`   ${chalk_1.default.gray('└')} tokenop execute <template-id> - Execute specific optimization`);
        }
        console.log(`   ${chalk_1.default.gray('└')} tokenop plan - Generate detailed implementation plan`);
        console.log(`   ${chalk_1.default.gray('└')} tokenop templates - Browse all available templates\n`);
        // Save results if requested
        if (options.output) {
            const results = {
                discovery_time: new Date().toISOString(),
                environment,
                matching_templates: matchingTemplates.map(t => t.id),
                optimization_plans: optimizationPlans,
                total_savings: totalSavings,
                total_implementation_cost: totalImplementationCost
            };
            await require('fs-extra').writeJson(options.output, results, { spaces: 2 });
            console.log(chalk_1.default.gray(`Results saved to ${options.output}\n`));
        }
    }
    catch (error) {
        spinner.fail('Discovery failed');
        console.error(chalk_1.default.red('\n❌ Error:'), error instanceof Error ? error.message : String(error));
        console.error(chalk_1.default.gray('\nFor debugging: tokenop --verbose discover\n'));
        process.exit(1);
    }
});
/**
 * Execute specific template command
 */
program
    .command('execute <template-id>')
    .description('🚀 Execute a specific optimization template')
    .option('--dry-run', 'Simulate execution without making changes')
    .option('--skip-prerequisites', 'Skip prerequisite validation (dangerous)')
    .action(async (templateId, options) => {
    console.log(chalk_1.default.blue.bold(`\n🚀 Executing Template: ${templateId}\n`));
    const spinner = (0, ora_1.default)('Preparing execution...').start();
    try {
        // Load template
        await templateEngine.loadTemplates();
        const template = templateEngine.getTemplate(templateId);
        if (!template) {
            spinner.fail('Template not found');
            console.error(chalk_1.default.red(`❌ Template '${templateId}' not found`));
            console.log(chalk_1.default.gray('\nAvailable templates:'));
            const allTemplates = templateEngine.listTemplates();
            allTemplates.forEach(t => console.log(`  • ${t.id} - ${t.name}`));
            return;
        }
        // Discover environment
        spinner.text = 'Discovering environment...';
        const environment = await discoveryAgent.discoverEnvironment();
        // Execute template
        spinner.text = `Executing ${template.name}...`;
        const result = await executionAgent.executeTemplate(template, environment, {
            dryRun: options.dryRun,
            skipPrerequisites: options.skipPrerequisites
        });
        spinner.succeed('Template execution complete');
        // Show results
        console.log(chalk_1.default.green.bold('\n✅ Execution Results:\n'));
        console.log(`Status: ${result.status}`);
        console.log(`Duration: ${((result.end_time?.getTime() || Date.now()) - result.start_time.getTime()) / 1000}s`);
        if (result.cost_savings) {
            console.log(`💰 Cost Savings: $${result.cost_savings.toLocaleString()}/month`);
        }
        if (result.roi_achieved) {
            console.log(`📈 ROI: ${result.roi_achieved.toFixed(1)}%`);
        }
        console.log(`Quality Preserved: ${result.quality_preserved ? '✅' : '❌'}`);
        if (result.steps_completed.length > 0) {
            console.log('\n📋 Steps Completed:');
            result.steps_completed.forEach(step => console.log(`  ✅ ${step}`));
        }
        if (result.steps_failed.length > 0) {
            console.log('\n❌ Failed Steps:');
            result.steps_failed.forEach(step => console.log(`  ❌ ${step}`));
        }
    }
    catch (error) {
        spinner.fail('Execution failed');
        console.error(chalk_1.default.red('\n❌ Error:'), error instanceof Error ? error.message : String(error));
        process.exit(1);
    }
});
/**
 * List templates command
 */
program
    .command('templates')
    .description('📋 List all available optimization templates')
    .option('--category <category>', 'Filter by category')
    .option('--detailed', 'Show detailed information')
    .action(async (options) => {
    console.log(chalk_1.default.blue.bold('\n📋 Available Optimization Templates\n'));
    try {
        await templateEngine.loadTemplates();
        let templates = templateEngine.listTemplates();
        if (options.category) {
            templates = templates.filter(t => t.category === options.category);
        }
        // Group by category
        const byCategory = templates.reduce((acc, template) => {
            if (!acc[template.category])
                acc[template.category] = [];
            acc[template.category].push(template);
            return acc;
        }, {});
        for (const [category, categoryTemplates] of Object.entries(byCategory)) {
            console.log(chalk_1.default.cyan.bold(`\n${category.toUpperCase()}:`));
            categoryTemplates.forEach(template => {
                console.log(`\n  ${chalk_1.default.blue(template.id)} - ${template.name}`);
                console.log(`    ${chalk_1.default.gray(template.description)}`);
                console.log(`    Confidence: ${chalk_1.default.yellow((template.confidence * 100).toFixed(1) + '%')} | ` +
                    `Success Count: ${template.success_count} | ` +
                    `Risk: ${template.optimization.risk_level}`);
                if (options.detailed) {
                    console.log(`    Expected Savings: ${template.optimization.expected_cost_reduction || 'Variable'}`);
                    console.log(`    Implementation: ${template.optimization.effort_estimate}`);
                }
            });
        }
        console.log(chalk_1.default.blue(`\n📊 Total: ${templates.length} templates available\n`));
    }
    catch (error) {
        console.error(chalk_1.default.red('❌ Error loading templates:'), error instanceof Error ? error.message : String(error));
        process.exit(1);
    }
});
/**
 * Analyze optimization economics for matched templates
 */
async function analyzeOptimizationEconomics(templates, environment) {
    const plans = [];
    for (const template of templates) {
        try {
            const baseline = await economicsCalculator.calculateBaseline(template, environment);
            const projected = await economicsCalculator.calculateProjectedSavings(template, baseline);
            const roi = economicsCalculator.calculateROI(baseline, projected, template.economics);
            plans.push({
                template,
                baseline,
                projected,
                projectedSavings: projected.monthly_savings || 0,
                roi,
                implementationCost: template.economics.implementation_cost.total_cost
            });
        }
        catch (error) {
            console.warn(`⚠️  Could not calculate economics for ${template.id}: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
    // Sort by ROI descending
    return plans.sort((a, b) => b.roi - a.roi);
}
// Program configuration
program
    .name('tokenop')
    .description('🔧 LLM inference cost optimization through template-driven Claude Code SDK agents')
    .version('0.1.0');
// Parse and execute
program.parse();
