/**
 * Professional Chatbot Widget for Sahibpreet Singh's Website
 * Client-side RAG system using Transformers.js for browser-based AI
 */

// Import Transformers.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

// Configure Transformers.js for browser use
env.allowRemoteModels = true;
env.allowLocalModels = true;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';

class SahibpreetChatbot {
    constructor(config = {}) {
        this.config = {
            position: config.position || 'bottom-right',
            theme: config.theme || 'auto', // 'light', 'dark', 'auto'
            placeholder: config.placeholder || 'Ask me about Sahibpreet Singh...',
            welcomeMessage: config.welcomeMessage || "👋 Hi! I can help you learn about Sahibpreet Singh's experience, skills, and projects. What would you like to know?",
            ...config
        };
        
        this.isOpen = false;
        this.conversationId = this.generateConversationId();
        this.isTyping = false;
        
        // RAG system components
        this.embeddingModel = null;
        this.qaModel = null;
        this.resumeChunks = [];
        this.embeddings = [];
        this.isInitialized = false;
        
        this.init();
    }
    
    async init() {
        this.createChatWidget();
        this.attachEventListeners();
        await this.initializeRAGSystem();
        this.loadSuggestions();
    }
    
    async initializeRAGSystem() {
        try {
            this.updateLoadingProgress('Loading embedding model...', 10);
            
            // Load embedding model for RAG
            this.embeddingModel = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
                progress_callback: (progress) => {
                    if (progress.status === 'downloading') {
                        const percent = Math.round((progress.loaded / progress.total) * 30) + 10;
                        this.updateLoadingProgress(`Downloading embedding model... ${percent}%`, percent);
                    }
                }
            });
            
            this.updateLoadingProgress('Loading QA model...', 50);
            
            // Load question-answering model
            this.qaModel = await pipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad', {
                progress_callback: (progress) => {
                    if (progress.status === 'downloading') {
                        const percent = Math.round((progress.loaded / progress.total) * 30) + 50;
                        this.updateLoadingProgress(`Downloading QA model... ${percent}%`, percent);
                    }
                }
            });
            
            this.updateLoadingProgress('Processing resume...', 80);
            
            // Load and process resume
            await this.loadAndProcessResume();
            
            this.updateLoadingProgress('Ready!', 100);
            
            // Hide loading and show chat
            setTimeout(() => {
                document.getElementById('loadingSection').style.display = 'none';
                document.getElementById('chat-window').style.display = 'flex';
                this.isInitialized = true;
                console.log('✅ RAG system initialized successfully');
            }, 1000);
            
        } catch (error) {
            console.error('❌ Failed to initialize RAG system:', error);
            this.updateLoadingProgress(`Error: ${error.message}. Check console for details.`, 0);
            
            // Show fallback mode after 3 seconds
            setTimeout(() => {
                this.enableFallbackMode();
            }, 3000);
        }
    }
    
    createChatWidget() {
        // Create widget container
        const widget = document.createElement('div');
        widget.id = 'sahibpreet-chatbot';
        widget.className = `chatbot-widget ${this.config.position}`;
        
        widget.innerHTML = `
            <!-- Loading Section -->
            <div class="loading-section" id="loadingSection">
                <div class="loading-spinner"></div>
                <p id="loadingText">Initializing AI models...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <!-- Chat Toggle Button -->
            <div class="chat-toggle" id="chat-toggle">
                <div class="chat-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8 10H16M8 14H13M12 22L7 17H4C3.44772 17 3 16.5523 3 4 3H20C20.5523 3 21 3.44772 21 4V16C21 16.5523 20.5523 17 20 17H15L12 22Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="close-icon" style="display: none;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
            </div>
            
            <!-- Chat Window -->
            <div class="chat-window" id="chat-window" style="display: none;">
                <!-- Header -->
                <div class="chat-header">
                    <div class="avatar">
                        <div class="avatar-placeholder">SS</div>
                    </div>
                    <div class="header-info">
                        <h3>Sahibpreet Singh</h3>
                        <p>GenAI Consultant Assistant</p>
                    </div>
                    <div class="header-status">
                        <div class="status-dot"></div>
                    </div>
                </div>
                
                <!-- Messages Container -->
                <div class="chat-messages" id="chat-messages">
                    <div class="message bot-message">
                        <div class="message-content">
                            ${this.config.welcomeMessage}
                        </div>
                        <div class="message-time">${this.formatTime(new Date())}</div>
                    </div>
                </div>
                
                <!-- Suggestions -->
                <div class="chat-suggestions" id="chat-suggestions">
                    <!-- Suggestions will be loaded here -->
                </div>
                
                <!-- Input Area -->
                <div class="chat-input-container">
                    <div class="chat-input-wrapper">
                        <input 
                            type="text" 
                            id="chat-input" 
                            placeholder="${this.config.placeholder}"
                            maxlength="500"
                        />
                        <button id="send-button" class="send-button">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                    <div class="powered-by">
                        Powered by 🤗 Transformers.js • Ask about experience, skills, projects
                    </div>
                </div>
            </div>
        `;
        
        // Add CSS
        this.addStyles();
        
        // Append to body
        document.body.appendChild(widget);
    }
    
    addStyles() {
        if (document.getElementById('sahibpreet-chatbot-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'sahibpreet-chatbot-styles';
        styles.textContent = `
            /* Chatbot Widget Styles */
            .chatbot-widget {
                position: fixed;
                z-index: 10000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .chatbot-widget.bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .chatbot-widget.bottom-left {
                bottom: 20px;
                left: 20px;
            }
            
            /* Loading Section */
            .loading-section {
                position: absolute;
                bottom: 80px;
                right: 0;
                width: 300px;
                background: white;
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                border: 1px solid #e5e7eb;
                text-align: center;
            }
            
            .loading-spinner {
                width: 32px;
                height: 32px;
                border: 3px solid #f3f4f6;
                border-top: 3px solid #2563eb;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .progress-bar {
                width: 100%;
                height: 8px;
                background: #f3f4f6;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 16px;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #2563eb, #8b5cf6);
                border-radius: 4px;
                transition: width 0.3s ease;
                width: 0%;
            }
            
            /* Toggle Button */
            .chat-toggle {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #2563eb, #8b5cf6);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
                transition: all 0.3s ease;
                color: white;
            }
            
            .chat-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
            }
            
            /* Chat Window */
            .chat-window {
                width: 380px;
                height: 600px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                position: absolute;
                bottom: 80px;
                right: 0;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                border: 1px solid #e5e7eb;
            }
            
            /* Dark mode support */
            @media (prefers-color-scheme: dark) {
                .chat-window {
                    background: #1f2937;
                    border-color: #374151;
                    color: #f9fafb;
                }
            }
            
            /* Header */
            .chat-header {
                padding: 16px 20px;
                background: linear-gradient(135deg, #2563eb, #8b5cf6);
                color: white;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .avatar-placeholder {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.2);
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 14px;
            }
            
            .header-info h3 {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }
            
            .header-info p {
                margin: 0;
                font-size: 12px;
                opacity: 0.9;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                background: #10b981;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            /* Messages */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            
            .message {
                max-width: 85%;
                animation: fadeInUp 0.3s ease;
            }
            
            .message-content {
                padding: 12px 16px;
                border-radius: 18px;
                font-size: 14px;
                line-height: 1.4;
                word-wrap: break-word;
            }
            
            .message-time {
                font-size: 11px;
                color: #6b7280;
                margin-top: 4px;
                padding: 0 8px;
            }
            
            .bot-message {
                align-self: flex-start;
            }
            
            .bot-message .message-content {
                background: #f3f4f6;
                color: #1f2937;
            }
            
            .user-message {
                align-self: flex-end;
            }
            
            .user-message .message-content {
                background: linear-gradient(135deg, #2563eb, #8b5cf6);
                color: white;
            }
            
            .user-message .message-time {
                text-align: right;
            }
            
            /* Suggestions */
            .chat-suggestions {
                padding: 0 16px 8px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .suggestion-chip {
                background: #f3f4f6;
                color: #374151;
                padding: 8px 12px;
                border-radius: 20px;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
                border: 1px solid transparent;
            }
            
            .suggestion-chip:hover {
                background: #e5e7eb;
                border-color: #2563eb;
            }
            
            /* Input */
            .chat-input-container {
                padding: 16px;
                border-top: 1px solid #e5e7eb;
                background: white;
            }
            
            .chat-input-wrapper {
                display: flex;
                gap: 8px;
                align-items: center;
            }
            
            #chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #d1d5db;
                border-radius: 24px;
                outline: none;
                font-size: 14px;
                transition: border-color 0.2s ease;
            }
            
            #chat-input:focus {
                border-color: #2563eb;
            }
            
            .send-button {
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, #2563eb, #8b5cf6);
                border: none;
                border-radius: 50%;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: transform 0.2s ease;
            }
            
            .send-button:hover {
                transform: scale(1.05);
            }
            
            .send-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .powered-by {
                text-align: center;
                font-size: 11px;
                color: #6b7280;
                margin-top: 8px;
            }
            
            /* Typing indicator */
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 12px 16px;
                background: #f3f4f6;
                border-radius: 18px;
                margin: 8px 0;
            }
            
            .typing-dot {
                width: 6px;
                height: 6px;
                background: #6b7280;
                border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
            }
            
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* Mobile responsive */
            @media (max-width: 768px) {
                .chat-window {
                    width: calc(100vw - 40px);
                    height: calc(100vh - 120px);
                    bottom: 80px;
                    right: 20px;
                }
                
                .chatbot-widget.bottom-left .chat-window {
                    left: 20px;
                    right: auto;
                }
            }
            
            /* Dark mode */
            @media (prefers-color-scheme: dark) {
                .loading-section {
                    background: #1f2937;
                    border-color: #374151;
                    color: #f9fafb;
                }
                
                .loading-spinner {
                    border-color: #374151;
                    border-top-color: #3b82f6;
                }
                
                .progress-bar {
                    background: #374151;
                }
                
                .bot-message .message-content {
                    background: #374151;
                    color: #f9fafb;
                }
                
                .suggestion-chip {
                    background: #374151;
                    color: #f9fafb;
                }
                
                .suggestion-chip:hover {
                    background: #4b5563;
                }
                
                .chat-input-container {
                    background: #1f2937;
                    border-color: #374151;
                }
                
                #chat-input {
                    background: #374151;
                    color: #f9fafb;
                    border-color: #4b5563;
                }
                
                .message-time {
                    color: #9ca3af;
                }
                
                .powered-by {
                    color: #9ca3af;
                }
            }
        `;
        
        document.head.appendChild(styles);
    }
    
    attachEventListeners() {
        const toggle = document.getElementById('chat-toggle');
        const sendButton = document.getElementById('send-button');
        const input = document.getElementById('chat-input');
        
        toggle.addEventListener('click', () => this.toggleChat());
        sendButton.addEventListener('click', () => this.sendMessage());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        // Auto-resize input
        input.addEventListener('input', () => {
            const isEmpty = input.value.trim() === '';
            sendButton.disabled = isEmpty;
        });
    }
    
    async loadAndProcessResume() {
        // Resume content
        const resumeContent = `# Sahibpreet Singh
**GenAI Consultant • Production ML Systems • $700K+ Project Impact**

## Contact Information
- **Email:** ss9334931@gmail.com
- **LinkedIn:** [linkedin.com/in/sahibpreetsinghh/](https://linkedin.com/in/sahibpreetsinghh/)
- **GitHub:** [github.com/sahibpreetsingh12](https://github.com/sahibpreetsingh12)
- **Kaggle:** [kaggle.com/sahib12](https://kaggle.com/sahib12)

## Professional Summary
GenAI Consultant at CGI with expertise in building production-scale AI systems that deliver real business value. Specialized in Agentic RAG systems, LLM evaluation frameworks, and custom CUDA kernel optimization. Delivered $700K+ project value and 65% efficiency improvements in recruitment processes.

## Current Role
**GenAI Consultant | CGI | 2024 - Present**
- Architected Agentic RAG systems resulting in $700K project win
- Developed Zero-Trust RAG system achieving 65% faster recruitment processes  
- Optimized Databricks + PySpark pipelines with 31% performance improvement
- Led cross-functional teams in implementing enterprise AI solutions

## Previous Experience

**GenAI Engineer | AI Talentflow | 2023 - 2024**
- Built Resume-2-ResumeRAG system generating $15K revenue increase
- Deployed production GenAI applications using Docker + AWS ECS

**Data Scientist | Tatras Data | 2022 - 2023**
- Developed Text-to-SQL systems with LLMs achieving 23% MAU growth
- Built ML transaction analysis models with 62% accuracy improvement
- Created contextual chatbots resulting in 128% revenue increase

**ML Engineer | ZS Associates | 2021 - 2022**
- Developed forecasting models generating $325K annual revenue
- Built pharma competition analysis using advanced SBERT/RoBERTa models

## Technical Skills

### AI/ML Frameworks
- PyTorch, Transformers, Langchain, LlamaIndex

### Agentic AI
- CrewAI, Langgraph, Autogen, SmolAgents

### Cloud Platforms
- Azure (Promptflow, ML Studio, Key Vault)
- AWS (ECS, Lambda, SageMaker)

### Data Engineering
- Databricks, PySpark, Neo4j, MongoDB, CosmosDB

### DevOps & Infrastructure
- Docker, Terraform, GitHub Actions, Kubernetes

### Programming Languages
- Python, CUDA, SQL, JavaScript

## Education
- **Post Graduate Certificate in AI & Machine Learning** | Lambton College | 2024-2025
- **Bachelor of Technology in Computer Science** | Punjab Technical University | 2017-2021

## Key Projects

### Zero-Trust RAG System
Enterprise AI security solution with semantic matching and Azure Key Vault integration

### Custom CUDA Kernels
Optimized GPU acceleration for LLM inference using Triton

### Resume-2-ResumeRAG
Production GenAI system with fine-tuned LLMs deployed on AWS ECS

### Tokenizer Fertility Research
Novel insights into subword optimization for production LLMs

## Research Focus
- Tokenizer fertility rates and their impact on LLM performance in production systems
- Zero-trust AI architectures for secure enterprise deployment
- Custom CUDA kernels for accelerated inference optimization
- Multi-agent RAG evaluation frameworks for production systems

## Key Achievements
- **$700K+ project value** delivered at CGI through innovative RAG systems
- **65% efficiency improvement** in recruitment processes using AI
- **31% ML pipeline optimization** using Databricks and PySpark
- **100+ engineers mentored** in AI/ML practices and best practices
- **IEEE Hackathon Winner (2nd Place)** for Explainable AI solution
- **Technical content creator** with growing audience on LinkedIn`;
        
        // Chunk the resume into semantic sections
        this.resumeChunks = this.chunkText(resumeContent);
        
        // Generate embeddings for each chunk
        for (let i = 0; i < this.resumeChunks.length; i++) {
            const embedding = await this.embeddingModel(this.resumeChunks[i].text);
            this.embeddings.push({
                index: i,
                embedding: embedding.data,
                text: this.resumeChunks[i].text,
                section: this.resumeChunks[i].section
            });
        }
    }
    
    chunkText(text) {
        const chunks = [];
        const sections = text.split(/\n## /);
        
        sections.forEach((section, index) => {
            if (section.trim()) {
                // Add back the ## for sections after the first
                const sectionText = index > 0 ? '## ' + section : section;
                const lines = sectionText.split('\n');
                const sectionTitle = lines[0].replace(/^#+\s*/, '');
                
                // For longer sections, split into smaller chunks
                if (sectionText.length > 500) {
                    const paragraphs = sectionText.split(/\n\n+/);
                    let currentChunk = '';
                    
                    paragraphs.forEach(paragraph => {
                        if (currentChunk.length + paragraph.length > 500 && currentChunk) {
                            chunks.push({
                                text: currentChunk.trim(),
                                section: sectionTitle
                            });
                            currentChunk = paragraph;
                        } else {
                            currentChunk += (currentChunk ? '\n\n' : '') + paragraph;
                        }
                    });
                    
                    if (currentChunk) {
                        chunks.push({
                            text: currentChunk.trim(),
                            section: sectionTitle
                        });
                    }
                } else {
                    chunks.push({
                        text: sectionText.trim(),
                        section: sectionTitle
                    });
                }
            }
        });
        
        return chunks;
    }
    
    async findRelevantChunks(question, topK = 3) {
        // Generate embedding for the question
        const questionEmbedding = await this.embeddingModel(question);
        
        // Calculate cosine similarity with all chunks
        const similarities = this.embeddings.map(chunk => ({
            ...chunk,
            similarity: this.cosineSimilarity(questionEmbedding.data, chunk.embedding)
        }));
        
        // Sort by similarity and return top K
        similarities.sort((a, b) => b.similarity - a.similarity);
        return similarities.slice(0, topK);
    }
    
    cosineSimilarity(a, b) {
        const dotProduct = a.reduce((sum, a_i, i) => sum + a_i * b[i], 0);
        const magnitudeA = Math.sqrt(a.reduce((sum, a_i) => sum + a_i * a_i, 0));
        const magnitudeB = Math.sqrt(b.reduce((sum, b_i) => sum + b_i * b_i, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }
    
    updateLoadingProgress(text, percent) {
        const loadingText = document.getElementById('loadingText');
        const progressFill = document.getElementById('progressFill');
        
        if (loadingText) loadingText.textContent = text;
        if (progressFill) progressFill.style.width = `${percent}%`;
    }
    
    enableFallbackMode() {
        // Hide loading and show chat with fallback functionality
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('chat-window').style.display = 'flex';
        this.isInitialized = true;
        
        // Add fallback message
        this.addMessage("⚠️ AI models couldn't load, but I can still help with basic questions about Sahibpreet Singh's resume using fallback responses.", 'bot');
        
        console.log('🔄 Chatbot running in fallback mode');
    }
    
    getFallbackResponse(message) {
        const messageLower = message.toLowerCase();
        
        // Simple keyword-based responses for fallback
        if (messageLower.includes('experience') || messageLower.includes('work') || messageLower.includes('job')) {
            return "Sahibpreet Singh is a GenAI Consultant at CGI with expertise in production-scale AI systems. He has delivered $700K+ project value and has previous experience at AI Talentflow, Tatras Data, and ZS Associates in ML/AI roles.";
        }
        
        if (messageLower.includes('skill') || messageLower.includes('technology') || messageLower.includes('programming')) {
            return "His technical skills include:\n• AI/ML: PyTorch, Transformers, Langchain, LlamaIndex\n• Agentic AI: CrewAI, Langgraph, Autogen\n• Cloud: Azure, AWS\n• Languages: Python, CUDA, SQL, JavaScript\n• Data: Databricks, PySpark, Neo4j, MongoDB";
        }
        
        if (messageLower.includes('project') || messageLower.includes('built') || messageLower.includes('develop')) {
            return "Key projects include:\n• Zero-Trust RAG System - Enterprise AI security solution\n• Custom CUDA Kernels - GPU acceleration for LLM inference\n• Resume-2-ResumeRAG - Production GenAI system on AWS ECS\n• Tokenizer Fertility Research - Novel LLM optimization insights";
        }
        
        if (messageLower.includes('education') || messageLower.includes('degree') || messageLower.includes('study')) {
            return "Education:\n• Post Graduate Certificate in AI & Machine Learning | Lambton College (2024-2025)\n• Bachelor of Technology in Computer Science | Punjab Technical University (2017-2021)";
        }
        
        if (messageLower.includes('achievement') || messageLower.includes('impact') || messageLower.includes('result')) {
            return "Key achievements:\n• $700K+ project value delivered at CGI\n• 65% efficiency improvement in recruitment processes\n• 31% ML pipeline optimization\n• 100+ engineers mentored\n• IEEE Hackathon Winner (2nd Place)";
        }
        
        // Default response
        return "I can help you learn about Sahibpreet Singh's professional background. Try asking about his experience, skills, projects, education, or achievements. For more detailed information, please check his resume or LinkedIn profile.";
    }
    
    async loadSuggestions() {
        // Default suggestions for the RAG system
        const defaultSuggestions = [
            "What is Sahibpreet's experience with GenAI?",
            "Tell me about his technical skills",
            "What projects has he worked on?"
        ];
        
        this.displaySuggestions(defaultSuggestions);
    }
    
    displaySuggestions(suggestions) {
        const container = document.getElementById('chat-suggestions');
        container.innerHTML = suggestions.map(suggestion => 
            `<div class="suggestion-chip" onclick="sahibpreetChatbot.sendSuggestion('${suggestion}')">${suggestion}</div>`
        ).join('');
    }
    
    toggleChat() {
        this.isOpen = !this.isOpen;
        const window = document.getElementById('chat-window');
        const chatIcon = document.querySelector('.chat-icon');
        const closeIcon = document.querySelector('.close-icon');
        
        if (this.isOpen) {
            window.style.display = 'flex';
            chatIcon.style.display = 'none';
            closeIcon.style.display = 'block';
            
            // Focus input
            setTimeout(() => {
                document.getElementById('chat-input').focus();
            }, 100);
        } else {
            window.style.display = 'none';
            chatIcon.style.display = 'block';
            closeIcon.style.display = 'none';
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message || !this.isInitialized) return;
        
        // Add user message
        this.addMessage(message, 'user');
        input.value = '';
        
        // Show typing indicator
        this.showTyping();
        
        // Check guardrails first
        const guardrailCheck = this.checkGuardrails(message);
        if (!guardrailCheck.isValid) {
            setTimeout(() => {
                this.hideTyping();
                this.addMessage(guardrailCheck.response, 'bot');
                document.getElementById('chat-suggestions').innerHTML = '';
            }, 1000);
            return;
        }
        
        try {
            // Check if RAG models are available
            if (this.embeddingModel && this.qaModel) {
                // RAG process
                const relevantChunks = await this.findRelevantChunks(message);
                
                if (relevantChunks.length === 0 || relevantChunks[0].similarity < 0.3) {
                    this.hideTyping();
                    this.addMessage("I can only answer questions about Sahibpreet Singh's professional background, skills, and experience. Please ask about his work, projects, or technical expertise.", 'bot');
                    document.getElementById('chat-suggestions').innerHTML = '';
                    return;
                }
                
                // Create context from relevant chunks
                const context = relevantChunks.map(chunk => chunk.text).join('\n\n');
                
                // Use QA model to generate response
                const qaResult = await this.qaModel({
                    question: message,
                    context: context
                });
                
                // Format and add response
                let response = qaResult.answer;
                
                // Add source attribution
                const uniqueSections = [...new Set(relevantChunks.map(chunk => chunk.section))];
                if (uniqueSections.length > 0) {
                    response += `\n\n*Source: ${uniqueSections.join(', ')}*`;
                }
                
                this.hideTyping();
                this.addMessage(response, 'bot');
                document.getElementById('chat-suggestions').innerHTML = '';
            } else {
                // Fallback mode - simple keyword matching
                const response = this.getFallbackResponse(message);
                this.hideTyping();
                this.addMessage(response, 'bot');
                document.getElementById('chat-suggestions').innerHTML = '';
            }
            
        } catch (error) {
            console.error('Message processing error:', error);
            this.hideTyping();
            this.addMessage("I'm having trouble processing your question right now. Please try rephrasing your question.", 'bot');
            document.getElementById('chat-suggestions').innerHTML = '';
        }
    }
    
    checkGuardrails(message) {
        const messageLower = message.toLowerCase().trim();
        
        // Block off-topic questions
        const blockedKeywords = [
            'weather', 'politics', 'religion', 'other people', 'someone else',
            'another person', 'family', 'relationship', 'dating', 'marriage',
            'illegal', 'harmful', 'dangerous', 'violence', 'personal life'
        ];
        
        if (blockedKeywords.some(keyword => messageLower.includes(keyword))) {
            return {
                isValid: false,
                response: "I can only answer questions about Sahibpreet Singh's professional background, skills, and experience. Please ask about his work, projects, or technical expertise."
            };
        }
        
        return { isValid: true };
    }
    
    // RAG-based response generation (removed external API dependency)
    
    sendSuggestion(suggestion) {
        const input = document.getElementById('chat-input');
        input.value = suggestion;
        this.sendMessage();
    }
    
    addMessage(content, type) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
            <div class="message-time">${this.formatTime(new Date())}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    showTyping() {
        if (this.isTyping) return;
        
        this.isTyping = true;
        const messagesContainer = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-message';
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    hideTyping() {
        this.isTyping = false;
        const typingMessage = document.querySelector('.typing-message');
        if (typingMessage) {
            typingMessage.remove();
        }
    }
    
    formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    generateConversationId() {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 11);
    }
}

// Initialize the chatbot when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create global instance with RAG system
    window.sahibpreetChatbot = new SahibpreetChatbot({
        position: 'bottom-right',
        theme: 'auto',
        autoStart: true
    });
});