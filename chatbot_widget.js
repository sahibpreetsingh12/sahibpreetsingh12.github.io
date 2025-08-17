/**
 * Professional Chatbot Widget for Sahibpreet Singh's Website
 * Integrates with the Python FastAPI backend
 */

class SahibpreetChatbot {
    constructor(config = {}) {
        this.config = {
            apiUrl: config.apiUrl || 'http://localhost:8000',
            position: config.position || 'bottom-right',
            theme: config.theme || 'auto', // 'light', 'dark', 'auto'
            placeholder: config.placeholder || 'Ask me about Sahibpreet Singh...',
            welcomeMessage: config.welcomeMessage || "👋 Hi! I can help you learn about Sahibpreet Singh's experience, skills, and projects. What would you like to know?",
            ...config
        };
        
        this.isOpen = false;
        this.conversationId = this.generateConversationId();
        this.isTyping = false;
        this.apiAvailable = null; // null = not checked, true = available, false = fallback mode
        
        this.init();
    }
    
    async init() {
        this.createChatWidget();
        this.attachEventListeners();
        await this.checkApiAvailability();
        this.loadSuggestions();
    }
    
    async checkApiAvailability() {
        try {
            const response = await fetch(`${this.config.apiUrl}/health`, {
                method: 'GET',
                timeout: 3000
            });
            
            if (response.ok) {
                this.apiAvailable = true;
                console.log('✅ Chatbot API is available');
            } else {
                this.apiAvailable = false;
                console.log('⚠️ Chatbot API not responding, using fallback mode');
            }
        } catch (error) {
            this.apiAvailable = false;
            console.log('⚠️ Chatbot API not available, using fallback mode');
        }
    }
    
    createChatWidget() {
        // Create widget container
        const widget = document.createElement('div');
        widget.id = 'sahibpreet-chatbot';
        widget.className = `chatbot-widget ${this.config.position}`;
        
        widget.innerHTML = `
            <!-- Chat Toggle Button -->
            <div class="chat-toggle" id="chat-toggle">
                <div class="chat-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8 10H16M8 14H13M12 22L7 17H4C3.44772 17 3 16.5523 3 16V4C3 3.44772 3.44772 3 4 3H20C20.5523 3 21 3.44772 21 4V16C21 16.5523 20.5523 17 20 17H15L12 22Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
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
                        Powered by AI • Ask about experience, skills, projects
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
    
    async loadSuggestions() {
        try {
            if (this.apiAvailable === true) {
                const response = await fetch(`${this.config.apiUrl}/chat/suggestions`);
                const data = await response.json();
                
                if (data.suggestions) {
                    this.displaySuggestions(data.suggestions.slice(0, 3));
                    return;
                }
            }
        } catch (error) {
            // Fall through to default suggestions
        }
        
        // Fallback suggestions
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
        
        if (!message) return;
        
        // Add user message
        this.addMessage(message, 'user');
        input.value = '';
        
        // Show typing indicator
        this.showTyping();
        
        // Step 1: Check guardrails first
        const guardrailCheck = this.checkGuardrails(message);
        if (!guardrailCheck.isValid) {
            setTimeout(() => {
                this.hideTyping();
                this.addMessage(guardrailCheck.response, 'bot');
                document.getElementById('chat-suggestions').innerHTML = '';
            }, 1000);
            return;
        }
        
        // Check if API is available, use fallback if not
        if (this.apiAvailable === false) {
            setTimeout(() => {
                this.hideTyping();
                this.addMessage(this.getFallbackResponse(message), 'bot');
                document.getElementById('chat-suggestions').innerHTML = '';
            }, 1000);
            return;
        }
        
        try {
            const response = await fetch(`${this.config.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.conversationId
                })
            });
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            const data = await response.json();
            
            // Hide typing indicator
            this.hideTyping();
            
            // Add bot response
            this.addMessage(data.response, 'bot');
            
            // Clear suggestions after first interaction
            document.getElementById('chat-suggestions').innerHTML = '';
            
        } catch (error) {
            // Fallback to local responses
            this.apiAvailable = false;
            this.hideTyping();
            this.addMessage(this.getFallbackResponse(message), 'bot');
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
    
    getFallbackResponse(message) {
        const messageLower = message.toLowerCase();
        
        // Greeting responses
        if (messageLower.includes('hello') || messageLower.includes('hi') || messageLower.includes('hey')) {
            return "Hello! I'm here to help you learn about Sahibpreet Singh's professional background. What would you like to know about his experience, skills, or projects?";
        }
        
        // Experience questions
        if (messageLower.includes('experience') || messageLower.includes('work') || messageLower.includes('job') || messageLower.includes('career')) {
            return `Sahibpreet Singh is currently a GenAI Consultant at CGI where he has delivered significant impact:

• Architected Agentic RAG systems resulting in $700K project wins
• Developed Zero-Trust RAG systems achieving 65% faster recruitment processes  
• Optimized ML pipelines with 31% performance improvements using Databricks + PySpark
• Led cross-functional teams implementing enterprise AI solutions

He previously worked at AI Talentflow, Tatras Data, and ZS Associates, consistently delivering high-impact AI/ML solutions.`;
        }
        
        // Skills questions
        if (messageLower.includes('skill') || messageLower.includes('technology') || messageLower.includes('technical') || messageLower.includes('programming')) {
            return `Sahibpreet Singh's technical expertise includes:

**AI/ML:** PyTorch, Transformers, Langchain, LlamaIndex
**Agentic AI:** CrewAI, Langgraph, Autogen, SmolAgents  
**Cloud:** Azure (Promptflow, ML Studio), AWS (ECS, Lambda, SageMaker)
**Data:** Databricks, PySpark, Neo4j, MongoDB
**DevOps:** Docker, Terraform, Kubernetes, GitHub Actions
**Languages:** Python, CUDA, SQL, JavaScript

He specializes in production-scale AI systems and custom CUDA kernel development.`;
        }
        
        // Education questions
        if (messageLower.includes('education') || messageLower.includes('degree') || messageLower.includes('study') || messageLower.includes('university')) {
            return `Sahibpreet Singh's educational background:

• **Post Graduate Certificate in AI & Machine Learning** - Lambton College (2024-2025)
• **Bachelor of Technology in Computer Science** - Punjab Technical University (2017-2021)

He's currently pursuing advanced AI/ML studies to stay at the forefront of rapidly evolving technologies.`;
        }
        
        // Projects questions
        if (messageLower.includes('project') || messageLower.includes('built') || messageLower.includes('developed')) {
            return `Sahibpreet Singh's key projects include:

• **Zero-Trust RAG System:** Enterprise AI security solution with Azure Key Vault integration
• **Custom CUDA Kernels:** GPU optimization for LLM inference using Triton
• **Resume-2-ResumeRAG:** Production GenAI system deployed on AWS ECS
• **Tokenizer Fertility Research:** Novel insights into subword optimization

These projects demonstrate his ability to deliver enterprise-scale AI solutions with real business value.`;
        }
        
        // Contact questions
        if (messageLower.includes('contact') || messageLower.includes('email') || messageLower.includes('reach') || messageLower.includes('linkedin')) {
            return `You can connect with Sahibpreet Singh through:

• **Email:** ss9334931@gmail.com
• **LinkedIn:** linkedin.com/in/sahibpreetsinghh/
• **GitHub:** github.com/sahibpreetsingh12  
• **Kaggle:** kaggle.com/sahib12

He's always open to discussing GenAI systems, production ML challenges, and collaboration opportunities.`;
        }
        
        // General about questions
        if (messageLower.includes('about') || messageLower.includes('who') || messageLower.includes('sahibpreet')) {
            return `Sahibpreet Singh is a GenAI Consultant at CGI specializing in production-scale AI systems. He has delivered over $700K in project value through Agentic RAG systems and achieved 65% efficiency improvements. 

Expert in PyTorch, Transformers, Azure ML, and custom CUDA kernels, he's known for building AI systems that scale beyond demos to real-world enterprise applications. Currently pursuing advanced AI/ML education and actively researching tokenizer optimization.`;
        }
        
        // Default fallback
        return `I'd be happy to help you learn about Sahibpreet Singh! You can ask me about:

• His work experience and current role at CGI
• Technical skills and expertise in AI/ML
• Educational background and certifications  
• Key projects and achievements
• How to contact him for opportunities

What specific aspect of his background interests you?`;
    }
    
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
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

// Initialize the chatbot when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Auto-detect API URL or use fallback
    const apiUrl = window.location.hostname === 'localhost' 
        ? 'http://localhost:8000' 
        : 'https://your-chatbot-api.herokuapp.com'; // Replace with your deployed API
    
    // Create global instance
    window.sahibpreetChatbot = new SahibpreetChatbot({
        apiUrl: apiUrl,
        position: 'bottom-right',
        theme: 'auto',
        autoStart: true
    });
});