#!/usr/bin/env python3
"""
Resume Processor
Extracts and processes content from Sahibpreet Singh's resume PDF using semantic section-based chunking.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Install with: pip install PyPDF2")
    PyPDF2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Install with: pip install PyMuPDF")
    fitz = None

logger = logging.getLogger(__name__)

class ResumeProcessor:
    """Processes resume PDF and creates semantic section-based chunks"""
    
    def __init__(self):
        self.name = "Sahibpreet Singh"
        self.role = "GenAI Consultant at CGI"
        
    async def process_resume(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process the resume PDF and create semantic chunks based on sections
        
        Args:
            pdf_path: Path to the resume PDF file
            
        Returns:
            Dictionary containing processed resume data with semantic chunks
        """
        logger.info(f"📄 Processing resume from: {pdf_path}")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        
        # Parse into semantic sections
        sections = self._parse_resume_sections(text)
        
        # Create semantic chunks (one per meaningful section/subsection)
        chunks = self._create_semantic_chunks(sections, text)
        
        # Extract key information
        key_info = self._extract_key_information(sections)
        
        return {
            'raw_text': text,
            'sections': sections,
            'chunks': chunks,
            'summary': key_info.get('summary', ''),
            'key_skills': key_info.get('skills', []),
            'experience_years': key_info.get('experience_years', 0),
            'education': key_info.get('education', []),
            'projects': key_info.get('projects', []),
            'achievements': key_info.get('achievements', [])
        }
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using available libraries"""
        
        if not pdf_path.exists():
            logger.warning(f"Resume PDF not found at: {pdf_path}, using mock data")
            return self._get_mock_resume_data()
        
        text = ""
        
        # Try PyMuPDF first (better text extraction)
        if fitz:
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                logger.info("✅ Text extracted using PyMuPDF")
                return text.strip()
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Fallback to PyPDF2
        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text()
                logger.info("✅ Text extracted using PyPDF2")
                return text.strip()
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # If no PDF library available or extraction fails, use mock data
        logger.warning("⚠️ Using mock resume data based on known information")
        return self._get_mock_resume_data()
    
    def _get_mock_resume_data(self) -> str:
        """Mock resume data based on your actual experience"""
        return """
        SAHIBPREET SINGH
        GenAI Consultant • Production ML Systems • $700K+ Project Impact
        
        CONTACT
        Email: ss9334931@gmail.com
        LinkedIn: linkedin.com/in/sahibpreetsinghh/
        GitHub: github.com/sahibpreetsingh12
        Kaggle: kaggle.com/sahib12
        
        PROFESSIONAL SUMMARY
        GenAI Consultant at CGI with expertise in building production-scale AI systems that deliver real business value. 
        Specialized in Agentic RAG systems, LLM evaluation frameworks, and custom CUDA kernel optimization.
        Delivered $700K+ project value and 65% efficiency improvements in recruitment processes.
        
        CURRENT ROLE
        GenAI Consultant | CGI | 2024 - Present
        • Architected Agentic RAG systems resulting in $700K project win
        • Developed Zero-Trust RAG system achieving 65% faster recruitment processes  
        • Optimized Databricks + PySpark pipelines with 31% performance improvement
        • Led cross-functional teams in implementing enterprise AI solutions
        
        PREVIOUS EXPERIENCE
        GenAI Engineer | AI Talentflow | 2023 - 2024
        • Built Resume-2-ResumeRAG system generating $15K revenue increase
        • Deployed production GenAI applications using Docker + AWS ECS
        
        Data Scientist | Tatras Data | 2022 - 2023
        • Developed Text-to-SQL systems with LLMs achieving 23% MAU growth
        • Built ML transaction analysis models with 62% accuracy improvement
        • Created contextual chatbots resulting in 128% revenue increase
        
        ML Engineer | ZS Associates | 2021 - 2022
        • Developed forecasting models generating $325K annual revenue
        • Built pharma competition analysis using advanced SBERT/RoBERTa models
        
        TECHNICAL SKILLS
        AI/ML Frameworks: PyTorch, Transformers, Langchain, LlamaIndex
        Agentic AI: CrewAI, Langgraph, Autogen, SmolAgents
        Cloud Platforms: Azure (Promptflow, ML Studio, Key Vault), AWS (ECS, Lambda, SageMaker)
        Data Engineering: Databricks, PySpark, Neo4j, MongoDB, CosmosDB
        DevOps: Docker, Terraform, GitHub Actions, Kubernetes
        Programming: Python, CUDA, SQL, JavaScript
        
        EDUCATION
        Post Graduate Certificate in AI & Machine Learning | Lambton College | 2024-2025
        Bachelor of Technology in Computer Science | Punjab Technical University | 2017-2021
        
        KEY PROJECTS
        Zero-Trust RAG System: Enterprise AI security solution with semantic matching and Azure Key Vault integration
        Custom CUDA Kernels: Optimized GPU acceleration for LLM inference using Triton
        Tokenizer Fertility Research: Novel insights into subword optimization for production LLMs
        Resume-2-ResumeRAG: Production GenAI system with fine-tuned LLMs deployed on AWS ECS
        
        RESEARCH FOCUS
        • Tokenizer fertility rates and their impact on LLM performance
        • Zero-trust AI architectures for enterprise deployment
        • Custom CUDA kernels for accelerated inference
        • Multi-agent RAG evaluation frameworks
        
        ACHIEVEMENTS
        • $700K+ project value delivered at CGI
        • 65% efficiency improvement in recruitment processes  
        • 31% ML pipeline optimization using Databricks
        • 100+ engineers mentored in AI/ML practices
        • IEEE Hackathon Winner (2nd Place) - Explainable AI
        • Technical content creator with growing LinkedIn audience
        """
    
    def _parse_resume_sections(self, text: str) -> Dict[str, str]:
        """Parse resume into logical sections"""
        sections = {}
        
        # Clean up the text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        current_section = "header"
        current_content = []
        
        # Define section headers (case insensitive)
        section_headers = {
            'contact': ['contact', 'contact information'],
            'summary': ['professional summary', 'summary', 'objective', 'profile'],
            'current_role': ['current role', 'present role'],
            'experience': ['experience', 'work experience', 'professional experience', 'previous experience'],
            'education': ['education', 'academic background', 'qualifications'],
            'skills': ['technical skills', 'skills', 'competencies', 'technologies'],
            'projects': ['key projects', 'projects', 'notable projects', 'major projects'],
            'research': ['research', 'research focus', 'publications', 'papers'],
            'achievements': ['achievements', 'accomplishments', 'awards', 'recognition']
        }
        
        for line in lines:
            line_upper = line.upper()
            
            # Check if this line is a section header
            is_section_header = False
            for section_key, headers in section_headers.items():
                for header in headers:
                    if header.upper() in line_upper and len(line) < 50:
                        # Save previous section
                        if current_content:
                            sections[current_section] = '\n'.join(current_content)
                        
                        # Start new section
                        current_section = section_key
                        current_content = []
                        is_section_header = True
                        break
                if is_section_header:
                    break
            
            # Add content to current section
            if not is_section_header:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _create_semantic_chunks(self, sections: Dict[str, str], full_text: str) -> List[Dict[str, str]]:
        """Create semantic chunks based on resume sections"""
        chunks = []
        
        # Basic info chunk
        contact_info = sections.get('contact', '') + '\n' + sections.get('header', '')
        if contact_info.strip():
            chunks.append({
                'content': f"Sahibpreet Singh is a GenAI Consultant at CGI. His contact information: {contact_info}",
                'type': 'contact',
                'metadata': {'section': 'contact_info'}
            })
        
        # Professional summary chunk
        summary = sections.get('summary', '')
        if summary:
            chunks.append({
                'content': f"About Sahibpreet Singh's professional background: {summary}",
                'type': 'summary',
                'metadata': {'section': 'professional_summary'}
            })
        
        # Current role chunk
        current_role = sections.get('current_role', '')
        if current_role:
            chunks.append({
                'content': f"Sahibpreet Singh's current role and responsibilities: {current_role}",
                'type': 'experience',
                'metadata': {'section': 'current_position'}
            })
        
        # Experience chunk
        experience = sections.get('experience', '')
        if experience:
            chunks.append({
                'content': f"Sahibpreet Singh's previous work experience: {experience}",
                'type': 'experience',
                'metadata': {'section': 'work_history'}
            })
        
        # Technical skills chunk
        skills = sections.get('skills', '')
        if skills:
            chunks.append({
                'content': f"Sahibpreet Singh's technical skills and expertise: {skills}",
                'type': 'skills',
                'metadata': {'section': 'technical_abilities'}
            })
        
        # Education chunk
        education = sections.get('education', '')
        if education:
            chunks.append({
                'content': f"Sahibpreet Singh's educational background: {education}",
                'type': 'education',
                'metadata': {'section': 'academic_background'}
            })
        
        # Projects chunk
        projects = sections.get('projects', '')
        if projects:
            chunks.append({
                'content': f"Sahibpreet Singh's key projects and implementations: {projects}",
                'type': 'projects',
                'metadata': {'section': 'notable_projects'}
            })
        
        # Research chunk
        research = sections.get('research', '')
        if research:
            chunks.append({
                'content': f"Sahibpreet Singh's research focus and interests: {research}",
                'type': 'research',
                'metadata': {'section': 'research_work'}
            })
        
        # Achievements chunk
        achievements = sections.get('achievements', '')
        if achievements:
            chunks.append({
                'content': f"Sahibpreet Singh's achievements and recognition: {achievements}",
                'type': 'achievements',
                'metadata': {'section': 'accomplishments'}
            })
        
        # Add a comprehensive overview chunk
        chunks.append({
            'content': f"""Sahibpreet Singh is a GenAI Consultant at CGI specializing in production-scale AI systems. 
            He has delivered $700K+ in project value through Agentic RAG systems, achieved 65% efficiency improvements, 
            and optimized ML pipelines by 31%. Expert in PyTorch, Transformers, Azure ML, Langchain, and custom CUDA kernels. 
            Known for building AI systems that scale beyond demos to real-world enterprise applications. 
            Currently pursuing advanced AI/ML education at Lambton College and actively researching tokenizer optimization.""",
            'type': 'overview',
            'metadata': {'section': 'complete_profile'}
        })
        
        return chunks
    
    def _extract_key_information(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract key structured information"""
        return {
            'summary': """GenAI Consultant at CGI specializing in production AI systems. 
                         $700K+ project impact, 65% efficiency improvements, expert in RAG systems and LLM optimization.""",
            'skills': ['PyTorch', 'Transformers', 'Langchain', 'Azure ML', 'CUDA', 'RAG Systems', 'LLM Optimization'],
            'experience_years': 3,
            'education': ['AI & ML Post Graduate Certificate - Lambton College', 'BTech Computer Science - Punjab Technical University'],
            'projects': ['Zero-Trust RAG System', 'Custom CUDA Kernels', 'Resume-2-ResumeRAG', 'Tokenizer Fertility Research'],
            'achievements': ['$700K project value', '65% efficiency improvement', '100+ engineers mentored', 'IEEE Hackathon Winner']
        }