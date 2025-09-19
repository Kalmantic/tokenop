"use strict";
/**
 * Template Engine - Loads and executes optimization templates
 * This is the core system that turns TokenSqueeze insights into actionable optimizations
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.TemplateEngine = void 0;
const yaml = __importStar(require("yaml"));
const fs = __importStar(require("fs-extra"));
const path = __importStar(require("path"));
const glob_1 = require("glob");
const https_1 = __importDefault(require("https"));
class TemplateEngine {
    templatesDirectory;
    templates = new Map();
    templatesLoaded = false;
    constructor(templatesDirectory) {
        this.templatesDirectory = templatesDirectory;
        // Default to design/templates or look for templates in standard locations
        this.templatesDirectory = templatesDirectory || this.findTemplatesDirectory();
    }
    /**
     * Stage 1: Configuration ⏳
     * Load all optimization templates from the design folder
     */
    async loadTemplates() {
        console.log("Stage 1: Template Loading ⏳");
        try {
            // Try local templates first
            await this.extractTemplatesFromDesignDoc();
            await this.loadStandaloneTemplateFiles();
            // If no local templates found, load from GitHub
            if (this.templates.size === 0) {
                console.log("  📡 No local templates found, loading from GitHub repository...");
                await this.loadTemplatesFromGitHub();
            }
            console.log(`Stage 1: Template Loading ✅ - Loaded ${this.templates.size} templates`);
            this.templatesLoaded = true;
        }
        catch (error) {
            console.error("Stage 1: Template Loading ❌", error);
            throw new Error(`Failed to load templates: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
    /**
     * Extract templates from the main design document
     * Parses the TokenOp Template v0.2.md file to extract YAML templates
     */
    async extractTemplatesFromDesignDoc() {
        // If templatesDirectory is the design folder, look directly in it
        // Otherwise, look in the parent directory (design folder)
        const templatesDir = this.templatesDirectory || '.';
        const designDocPath = templatesDir.endsWith('design')
            ? path.join(templatesDir, 'TokenOp Template v0.2.md')
            : path.join(templatesDir, '..', 'TokenOp Template v0.2.md');
        if (!await fs.pathExists(designDocPath)) {
            console.log("Design document not found, looking for standalone templates...");
            return;
        }
        const content = await fs.readFile(designDocPath, 'utf-8');
        // Extract YAML blocks from markdown
        const yamlBlocks = this.extractYamlFromMarkdown(content);
        console.log(`Found ${yamlBlocks.length} YAML templates in design document`);
        for (const yamlBlock of yamlBlocks) {
            try {
                const template = yaml.parse(yamlBlock);
                if (template.id && template.name) {
                    this.templates.set(template.id, template);
                    console.log(`  ✅ Loaded template: ${template.id} - ${template.name}`);
                }
            }
            catch (error) {
                console.warn(`  ⚠️  Failed to parse template YAML: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
    }
    /**
     * Extract YAML blocks from markdown content
     */
    extractYamlFromMarkdown(content) {
        const yamlBlocks = [];
        const lines = content.split('\n');
        let inYamlBlock = false;
        let currentBlock = [];
        for (const line of lines) {
            if (line.trim() === '```' && inYamlBlock) {
                // End of YAML block
                if (currentBlock.length > 0) {
                    const yamlContent = currentBlock.join('\n');
                    // Only include blocks that look like template YAML (have id and name)
                    if (yamlContent.includes('id:') && yamlContent.includes('name:')) {
                        yamlBlocks.push(yamlContent);
                    }
                }
                inYamlBlock = false;
                currentBlock = [];
            }
            else if (line.trim() === '---' && !inYamlBlock) {
                // Start of potential YAML block
                inYamlBlock = true;
                currentBlock = ['---'];
            }
            else if (inYamlBlock) {
                currentBlock.push(line);
            }
        }
        return yamlBlocks;
    }
    /**
     * Load standalone template files (if any exist)
     */
    async loadStandaloneTemplateFiles() {
        if (!await fs.pathExists(this.templatesDirectory || '.')) {
            return;
        }
        const templateFiles = await (0, glob_1.glob)('**/*.{yaml,yml}', {
            cwd: this.templatesDirectory || '.',
            absolute: true
        });
        for (const templateFile of templateFiles) {
            try {
                const content = await fs.readFile(templateFile, 'utf-8');
                const template = yaml.parse(content);
                if (template.id && template.name) {
                    this.templates.set(template.id, template);
                    console.log(`  ✅ Loaded standalone template: ${template.id}`);
                }
            }
            catch (error) {
                console.warn(`  ⚠️  Failed to load template ${templateFile}: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
    }
    /**
     * Stage 2: Template Matching ⏳
     * Find templates that match the detected environment
     */
    async findMatchingTemplates(environment) {
        console.log("Stage 2: Template Matching ⏳");
        if (!this.templatesLoaded) {
            await this.loadTemplates();
        }
        const matchingTemplates = [];
        for (const template of this.templates.values()) {
            const matchScore = this.calculateMatchScore(template, environment);
            if (matchScore > 0.7) { // 70% match threshold
                console.log(`  ✅ Template match: ${template.id} (score: ${matchScore.toFixed(2)})`);
                matchingTemplates.push(template);
            }
        }
        // Sort by confidence score and match quality
        matchingTemplates.sort((a, b) => {
            return (b.confidence * 100) - (a.confidence * 100);
        });
        console.log(`Stage 2: Template Matching ✅ - Found ${matchingTemplates.length} matching templates`);
        return matchingTemplates;
    }
    /**
     * Calculate how well a template matches the environment
     * Returns score 0-1 (higher = better match)
     */
    calculateMatchScore(template, environment) {
        const match = template.environment_match;
        let score = 0;
        let criteria = 0;
        // Application Layer Matching
        if (match.runtime) {
            criteria++;
            const runtimes = Array.isArray(match.runtime) ? match.runtime : [match.runtime];
            if (environment.application.runtime_detected.some(r => runtimes.includes(r))) {
                score++;
            }
        }
        if (match.batch_size) {
            criteria++;
            const currentBatchSize = environment.serving.performance_metrics.batch_efficiency;
            if (this.matchesRange(currentBatchSize, match.batch_size)) {
                score++;
            }
        }
        if (match.gpu_utilization) {
            criteria++;
            const utilization = environment.infrastructure.gpu_inventory[0]?.utilization || 0;
            if (this.matchesRange(utilization, match.gpu_utilization)) {
                score++;
            }
        }
        if (match.memory_bound !== undefined) {
            criteria++;
            const memoryEfficiency = environment.infrastructure.memory_analysis.bandwidth_efficiency;
            const isMemoryBound = memoryEfficiency < 0.3; // Low efficiency indicates memory bound
            if (match.memory_bound === isMemoryBound) {
                score++;
            }
        }
        // If no criteria specified, it's a universal template
        if (criteria === 0) {
            return 0.5;
        }
        return score / criteria;
    }
    /**
     * Check if a value matches a range specification
     * Range can be like "<30%", ">1000", "7B-30B", etc.
     */
    matchesRange(value, range) {
        const ranges = Array.isArray(range) ? range : [range];
        for (const r of ranges) {
            if (typeof r === 'string') {
                // Handle ranges like "<30%", ">1000", "7B", etc.
                const numValue = typeof value === 'string' ? parseFloat(value) : value;
                if (r.startsWith('<')) {
                    const threshold = parseFloat(r.substring(1).replace('%', ''));
                    if (numValue < threshold)
                        return true;
                }
                else if (r.startsWith('>')) {
                    const threshold = parseFloat(r.substring(1).replace('%', ''));
                    if (numValue > threshold)
                        return true;
                }
                else if (r.includes('-')) {
                    const [min, max] = r.split('-').map(x => parseFloat(x.replace(/[GB%]/g, '')));
                    if (numValue >= min && numValue <= max)
                        return true;
                }
                else {
                    // Exact match or contains
                    if (value.toString().includes(r) || r.includes(value.toString()))
                        return true;
                }
            }
        }
        return false;
    }
    /**
     * Get template by ID
     */
    getTemplate(templateId) {
        return this.templates.get(templateId);
    }
    /**
     * List all available templates
     */
    listTemplates() {
        return Array.from(this.templates.values());
    }
    /**
     * Get templates by category
     */
    getTemplatesByCategory(category) {
        return Array.from(this.templates.values()).filter(t => t.category === category);
    }
    /**
     * Find templates directory
     */
    findTemplatesDirectory() {
        const possiblePaths = [
            path.join(process.cwd(), 'design'),
            path.join(process.cwd(), 'design', 'templates'),
            path.join(process.cwd(), 'templates'),
            path.join(__dirname, '..', '..', 'design'),
            path.join(__dirname, '..', '..', 'design', 'templates')
        ];
        for (const possiblePath of possiblePaths) {
            if (fs.pathExistsSync(possiblePath)) {
                return possiblePath;
            }
        }
        // Default to design directory
        return path.join(process.cwd(), 'design');
    }
    /**
     * Validate template format
     */
    validateTemplate(template) {
        const errors = [];
        // Required fields
        if (!template.id)
            errors.push("Template missing required field: id");
        if (!template.name)
            errors.push("Template missing required field: name");
        if (!template.category)
            errors.push("Template missing required field: category");
        if (!template.optimization)
            errors.push("Template missing required field: optimization");
        if (!template.economics)
            errors.push("Template missing required field: economics");
        if (!template.implementation)
            errors.push("Template missing required field: implementation");
        // Validate confidence score
        if (template.confidence !== undefined && (template.confidence < 0 || template.confidence > 1)) {
            errors.push("Template confidence must be between 0 and 1");
        }
        // Validate implementation steps
        if (template.implementation && template.implementation.automated_steps) {
            for (const step of template.implementation.automated_steps) {
                if (!step.step_id)
                    errors.push(`Step missing step_id: ${step.name}`);
                if (!step.validation)
                    errors.push(`Step missing validation: ${step.step_id}`);
                if (!step.validation.rollback_command) {
                    errors.push(`Step missing rollback_command: ${step.step_id}`);
                }
            }
        }
        return {
            valid: errors.length === 0,
            errors
        };
    }
    /**
     * Load templates from GitHub repository
     */
    async loadTemplatesFromGitHub() {
        const githubUrl = 'https://raw.githubusercontent.com/Kalmantic/tokenop/main/design/TokenOp%20Template%20v0.2.md';
        try {
            const content = await this.fetchFromGitHub(githubUrl);
            console.log("  ✅ Downloaded template document from GitHub");
            // Extract YAML blocks from the downloaded content
            const yamlBlocks = this.extractYamlFromMarkdown(content);
            console.log(`  📋 Found ${yamlBlocks.length} potential templates in remote document`);
            for (const yamlBlock of yamlBlocks) {
                try {
                    const template = yaml.parse(yamlBlock);
                    if (template.id && template.name) {
                        this.templates.set(template.id, template);
                        console.log(`  ✅ Loaded remote template: ${template.id} - ${template.name}`);
                    }
                }
                catch (error) {
                    console.warn(`  ⚠️  Failed to parse remote template YAML: ${error instanceof Error ? error.message : String(error)}`);
                }
            }
        }
        catch (error) {
            console.warn(`  ⚠️  Failed to load templates from GitHub: ${error instanceof Error ? error.message : String(error)}`);
            throw error;
        }
    }
    /**
     * Fetch content from GitHub using HTTPS
     */
    async fetchFromGitHub(url) {
        return new Promise((resolve, reject) => {
            https_1.default.get(url, (response) => {
                if (response.statusCode !== 200) {
                    reject(new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`));
                    return;
                }
                let data = '';
                response.on('data', (chunk) => {
                    data += chunk;
                });
                response.on('end', () => {
                    resolve(data);
                });
            }).on('error', (error) => {
                reject(error);
            });
        });
    }
}
exports.TemplateEngine = TemplateEngine;
