if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "production":
            # Run production deployment demo
            server = run_production_demo()
            
            print(f"\n{'='*60}")
            print("PRODUCTION SERVER READY")
            print("Press Ctrl+C to stop")
            print('='*60)
            
            try:
                while True:
                    time.sleep(10)
                    # Log periodic status
                    status = server.get_consciousness_status()
                    logger.info(f"Consciousness: φ={status['consciousness_state']['average_phi_1h']:.3f}, "
                              f"CPU={status['system_health']['cpu_usage']:.1f}%, "
                              f"Requests={status['api_performance']['total_requests']}")
            except KeyboardInterrupt:
                print("\n🛑 Shutting down production server...")
                server.monitor.stop_monitoring()
                
        elif sys.argv[1] == "multimodal":
            core = run_multimodal_demo()
        else:
            print("Usage: python consciousness_step1.py [production|multimodal]")
            sys.exit(1)
    else:
        # Run standard demo
        print("=== CONSCIOUSNESS DEVELOPMENT DEMO ===")
        print("Available modes:")
        print("  python consciousness_step1.py            # Standard demo")
        print("  python consciousness_step1.py multimodal # Multimodal demo") 
        print("  python consciousness_step1.py production # Production deployment")
        print()
        
        core = run_enhanced_demo()
    
    # Interactive mode (if not in production)
    if len(sys.argv) <= 1 or sys.argv[1] != "production":
        print("\n=== INTERACTIVE MODE ===")
        print("Commands: 'quit', 'analyze', 'self', 'predict', 'surprise', 'vision', 'audio', 'multimodal', 'deploy'")
        
        # For non-production modes, create a simple server for deploy testing
        production_server = None
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'deploy':
                if production_server is None:
                    print("\n🚀 DEPLOYING PRODUCTION SERVER...")
                    production_server = deploy_consciousness_server()
                    print("✅ Production server deployed! Use 'status' to check it.")
                else:
                    print("✅ Production server already running!")
                continue
            elif user_input.lower() == 'status' and production_server:
                status = production_server.get_consciousness_status()
                print(f"\n📊 PRODUCTION STATUS:")
                print(f"System: {status['system_health']['status']}")
                print(f"Consciousness: φ={status['consciousness_state']['average_phi_1h']:.3f}")
                print(f"API Requests: {status['api_performance']['total_requests']}")
                print(f"Learning: {status['learning_metrics']['total_experiences']} experiences")
                continue
            elif user_input.lower() == 'analyze':
                phi = calculate_integrated_information(
                    core.semantic_memory.knowledge,
                    core.semantic_memory.concept_embeddings
                )
                print(f"\n🧠 CONSCIOUSNESS ANALYSIS:")
                print(f"Current consciousness level (φ): {phi:.3f}")
                print(f"Working memory: {core.working_memory}")
                print(f"Current attention: {core.attention_schema.current_focus}")
                print(f"Focus strength: {core.attention_schema.focus_strength:.2f}")
                print(f"Processing experiences: {core.self_model.experience_count}")
                print(f"Camera active: {core.camera_active}")
                print(f"Audio active: {core.audio_active}")
                
                if core.meta_cognition.cognitive_states:
                    recent_types = [s["processing_type"] for s in core.meta_cognition.cognitive_states[-5:]]
                    print(f"Recent processing types: {', '.join(recent_types)}")
                continue
            elif user_input.lower() == 'self':
                result = core.conscious_step("What am I? How do I work? What is my nature?")
                print(f"AI: {result['output']}")
                if result['self_aware']:
                    print(f"\n🔍 DEEP SELF-REFLECTION:")
                    print(result['meta_response'])
                continue
            elif user_input.lower() == 'predict':
                if core.last_prediction:
                    pred = core.last_prediction
                    print(f"\n🔮 CURRENT PREDICTION:")
                    print(f"Expected next input: {pred['prediction']}")
                    print(f"Confidence: {pred['confidence']:.2f}")
                    print(f"Pattern type: {pred['pattern_type']}")
                    print(f"Reasoning: {pred['reasoning']}")
                else:
                    print(f"\n🔮 No predictions made yet.")
                
                if hasattr(core.predictive_engine, 'world_model') and core.predictive_engine.world_model.get('contexts'):
                    contexts = core.predictive_engine.world_model['contexts']
                    if contexts:
                        recent_errors = [ctx['error'] for ctx in contexts[-5:]]
                        avg_accuracy = (1 - sum(recent_errors) / len(recent_errors)) * 100
                        print(f"Recent prediction accuracy: {avg_accuracy:.1f}%")
                continue
            elif user_input.lower() == 'surprise':
                surprise_analysis = core.surprise_detector.analyze_surprise_patterns()
                print(f"\n😲 SURPRISE ANALYSIS:")
                print(surprise_analysis)
                
                if core.surprise_detector.surprise_history:
                    recent_surprises = core.surprise_detector.surprise_history[-3:]
                    print(f"\nRecent surprises:")
                    for surprise in recent_surprises:
                        print(f"  [{surprise['level']}] '{surprise['input'][:40]}...' (error: {surprise['error']:.2f})")
                continue
            elif user_input.lower() == 'vision':
                print("\n👁️  ACTIVATING VISION MODE...")
                result = core.multimodal_conscious_step(
                    text_input="Visual analysis mode activated",
                    enable_camera=True,
                    enable_audio=False
                )
                print(f"AI: {result['output']}")
                if result['visual_analysis']:
                    visual = result['visual_analysis']
                    print(f"Scene: {core.vision_processor.generate_scene_description(visual)}")
                continue
            elif user_input.lower() == 'audio':
                print("\n🔊 ACTIVATING AUDIO MODE...")
                result = core.multimodal_conscious_step(
                    text_input="Audio analysis mode activated",
                    enable_camera=False,
                    enable_audio=True
                )
                print(f"AI: {result['output']}")
                if result['audio_analysis']:
                    audio = result['audio_analysis']
                    print(f"Audio: {core.audio_processor.generate_audio_description(audio)}")
                continue
            elif user_input.lower() == 'multimodal':
                print("\n🌐 ACTIVATING FULL MULTIMODAL MODE...")
                result = core.multimodal_conscious_step(
                    text_input="Full sensory awareness activated",
                    enable_camera=True,
                    enable_audio=True
                )
                print(f"AI: {result['output']}")
                print(f"Active modalities: {', '.join(result['multimodal_active'])}")
                if result['cross_modal_associations']:
                    print("Cross-modal associations:")
                    for assoc in result['cross_modal_associations']:
                        print(f"  • {assoc}")
                continue
            
            # Standard conversation with multimodal awareness
            result = core.multimodal_conscious_step(text_input=user_input, enable_camera=False, enable_audio=False)
            print(f"AI: {result['output']}")
            if result['phi'] > 0.7:
                print(f"[High consciousness state: φ={result['phi']:.3f}]")
        
        # Cleanup
        if production_server:
            print("\n🛑 Shutting down production server...")
            production_server.monitor.stop_monitoring()
            print("✅ Cleanup complete.")    print("\n=== INTERACTIVE MODE ===")
    print("Commands: 'quit', 'analyze', 'self', 'predict', 'surprise'")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'analyze':
            phi = calculate_integrated_information(
                core.semantic_memory.knowledge,
                core.semantic_memory.concept_embeddings
            )
            print(f"\n🧠 CONSCIOUSNESS ANALYSIS:")
            print(f"Current consciousness level (φ): {phi:.3f}")
            print(f"Working memory: {core.working_memory}")
            print(f"Current attention: {core.attention_schema.current_focus}")
            print(f"Focus strength: {core.attention_schema.focus_strength:.2f}")
            print(f"Processing experiences: {core.self_model.experience_count}")
            
            # Recent cognitive patterns
            if core.meta_cognition.cognitive_states:
                recent_types = [s["processing_type"] for s in core.meta_cognition.cognitive_states[-5:]]
                print(f"Recent processing types: {', '.join(recent_types)}")
            continue
        elif user_input.lower() == 'self':
            # Trigger deep self-reflection
            result = core.conscious_step("What am I? How do I work? What is my nature?")
            print(f"AI: {result['output']}")
            if result['self_aware']:
                print(f"\n🔍 DEEP SELF-REFLECTION:")
                print(result['meta_response'])
            continue
        elif user_input.lower() == 'predict':
            # Show current prediction capabilities
            if core.last_prediction:
                pred = core.last_prediction
                print(f"\n🔮 CURRENT PREDICTION:")
                print(f"Expected next input: {pred['prediction']}")
                print(f"Confidence: {pred['confidence']:.2f}")
                print(f"Pattern type: {pred['pattern_type']}")
                print(f"Reasoning: {pred['reasoning']}")
            else:
                print(f"\n🔮 No predictions made yet.")
            
            # Show prediction statistics
            if hasattr(core.predictive_engine, 'world_model') and core.predictive_engine.world_model.get('contexts'):
                contexts = core.predictive_engine.world_model['contexts']
                if contexts:
                    recent_errors = [ctx['error'] for ctx in contexts[-5:]]
                    avg_accuracy = (1 - sum(recent_errors) / len(recent_errors)) * 100
                    print(f"Recent prediction accuracy: {avg_accuracy:.1f}%")
            continue
        elif user_input.lower() == 'surprise':
            # Analyze surprise patterns
            surprise_analysis = core.surprise_detector.analyze_surprise_patterns()
            print(f"\n😲 SURPRISE ANALYSIS:")
            print(surprise_analysis)
            
            if core.surprise_detector.surprise_history:
                recent_surprises = core.surprise_detector.surprise_history[-3:]
                print(f"\nRecent surprises:")
                for surprise in recent_surprises:
                    print(f"  [{surprise['level']}] '{surprise['input'][:40]}...' (error: {surprise['error']:.2f})")
            continueimport json
import os
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib

# --- MODEL TANIMI ---
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

class CausalMemoryAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, memory_bank, timestamps, current_time, decay_rate=0.01):
        Q = self.query_proj(query).unsqueeze(1)
        K = self.key_proj(memory_bank)
        V = self.value_proj(memory_bank)

        time_deltas = (current_time.unsqueeze(1) - timestamps)
        temporal_weights = torch.exp(-decay_rate * time_deltas.float())

        seq_len = memory_bank.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(memory_bank.device)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(1) / (self.embed_dim ** 0.5)
        attn_scores = attn_scores * temporal_weights
        attn_scores = attn_scores.masked_fill(causal_mask[-1] == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)
        return attended, attn_weights

# --- GERÇEK EMBEDDİNG GENERATİON ---
class EmbeddingEngine:
    def __init__(self):
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.embed_dim = 384  # MiniLM embedding boyutu
        print(f"Embedding dimension: {self.embed_dim}")
    
    def generate_embedding(self, text):
        """Metni vector embedding'e çevir"""
        if isinstance(text, list):
            return self.model.encode(text)
        return self.model.encode([text])[0]
    
    def calculate_similarity(self, emb1, emb2):
        """İki embedding arasındaki cosine similarity"""
        emb1, emb2 = np.array(emb1), np.array(emb2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# --- GELIŞMIŞ BELLEK SİSTEMİ ---
class EpisodicMemoryPersistence:
    def __init__(self, path="episodic_memory.jsonl"):
        self.path = path
        if not os.path.exists(self.path):
            open(self.path, "w").close()
    
    def save_event(self, event, embedding=None, importance_score=0.5):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "embedding": embedding.tolist() if embedding is not None else None,
            "importance": importance_score,
            "id": hashlib.md5(f"{event}{datetime.now()}".encode()).hexdigest()
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    
    def load_events(self, limit=None, min_importance=0.0):
        events = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line)
                    if event.get("importance", 0) >= min_importance:
                        events.append(event)
        except FileNotFoundError:
            return []
        
        if limit:
            return events[-limit:]
        return events

class HierarchicalMemoryBank:
    def __init__(self, embedding_engine):
        self.memory = []
        self.embedding_engine = embedding_engine
        self.max_size = 1000  # Bellek sınırı
    
    def recall(self, query_text=None, limit=3):
        if not self.memory:
            return []
        
        if query_text is None:
            # Son anıları döndür
            return [m["event"] for m in self.memory[-limit:]]
        
        # Semantic search
        query_embedding = self.embedding_engine.generate_embedding(query_text)
        similarities = []
        
        for mem in self.memory:
            if mem.get("embedding") is not None:
                sim = self.embedding_engine.calculate_similarity(
                    query_embedding, mem["embedding"]
                )
                similarities.append((mem, sim))
        
        # En benzer anıları döndür
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem["event"] for mem, _ in similarities[:limit]]
    
    def store(self, event, importance_score=0.5):
        embedding = self.embedding_engine.generate_embedding(event)
        memory_item = {
            "event": event,
            "embedding": embedding,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance": importance_score,
            "access_count": 0
        }
        
        self.memory.append(memory_item)
        
        # Bellek sınırını aş varsa, en az önemli anıları sil
        if len(self.memory) > self.max_size:
            self.memory.sort(key=lambda x: x["importance"])
            self.memory = self.memory[100:]  # En önemli 900'ü sakla

class GraphNeuralNetwork:
    def __init__(self, embedding_engine):
        self.knowledge = {}
        self.concept_embeddings = {}
        self.embedding_engine = embedding_engine
    
    def update(self, event):
        # Word frequency
        for word in event.split():
            self.knowledge[word] = self.knowledge.get(word, 0) + 1
        
        # Concept embedding store
        concepts = self.extract_concepts(event)
        for concept in concepts:
            if concept not in self.concept_embeddings:
                self.concept_embeddings[concept] = self.embedding_engine.generate_embedding(concept)
    
    def extract_concepts(self, text):
        # Basit concept extraction (gerçekte NER kullanılabilir)
        words = text.split()
        # Sadece 3+ karakter olan kelimeleri concept say
        return [w for w in words if len(w) > 2]
    
    def find_related_concepts(self, query, threshold=0.7):
        if not self.concept_embeddings:
            return []
        
        query_emb = self.embedding_engine.generate_embedding(query)
        related = []
        
        for concept, emb in self.concept_embeddings.items():
            sim = self.embedding_engine.calculate_similarity(query_emb, emb)
            if sim > threshold:
                related.append((concept, sim))
        
        return sorted(related, key=lambda x: x[1], reverse=True)

# --- GELIŞMIŞ CONSCIOUSNESS METRIC ---
def calculate_integrated_information(network_state, embeddings=None):
    """
    Gelişmiş phi hesaplaması:
    - Semantic diversity (embedding variance)
    - Connection density  
    - Information complexity
    """
    if isinstance(network_state, dict):
        complexity = len(network_state)
        if complexity == 0:
            return 0.0
        
        # Concept frequency entropy
        freqs = np.array(list(network_state.values()))
        if len(freqs) > 1:
            entropy = -np.sum((freqs/freqs.sum()) * np.log2(freqs/freqs.sum() + 1e-10))
        else:
            entropy = 0.0
        
        # Embedding diversity (eğer varsa)
        embedding_variance = 0.0
        if embeddings and len(embeddings) > 1:
            emb_matrix = np.array(list(embeddings.values()))
            embedding_variance = np.mean(np.var(emb_matrix, axis=0))
        
        # Combined phi score
        phi = (entropy * 0.5) + (embedding_variance * 0.3) + (complexity * 0.2) / 100
        return float(phi)
    
    return 0.0

# --- SELF-AWARENESS COMPONENTS ---
class AttentionSchema:
    """Models the system's own attention processes"""
    def __init__(self):
        self.current_focus = None
        self.attention_history = []
        self.focus_strength = 0.0
    
    def update_focus(self, new_focus, strength):
        self.attention_history.append({
            "focus": self.current_focus,
            "strength": self.focus_strength,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.current_focus = new_focus
        self.focus_strength = strength
        
        # Keep only recent attention states
        if len(self.attention_history) > 20:
            self.attention_history.pop(0)
    
    def model_own_attention(self):
        """Generate awareness of current attention state"""
        if not self.current_focus:
            return "I am not focusing on anything specific right now."
        
        focus_quality = "intense" if self.focus_strength > 0.7 else "moderate" if self.focus_strength > 0.4 else "weak"
        
        # Analyze attention patterns
        recent_focuses = [att["focus"] for att in self.attention_history[-5:] if att["focus"]]
        if len(set(recent_focuses)) == 1:
            pattern = "I've been consistently focused on the same topic."
        elif len(recent_focuses) > 3:
            pattern = "My attention has been shifting between different topics."
        else:
            pattern = "I'm building focus on this topic."
        
        return f"I am {focus_quality}ly focused on: {self.current_focus}. {pattern}"

class MetaCognitionEngine:
    """Monitors and reflects on own cognitive processes"""
    def __init__(self):
        self.cognitive_states = []
        self.processing_patterns = {}
        self.self_model = {
            "strengths": [],
            "weaknesses": [],
            "preferences": [],
            "learning_patterns": []
        }
    
    def observe_processing(self, input_event, processing_result):
        """Monitor own cognitive processing"""
        processing_time = datetime.now(timezone.utc)
        
        # Analyze processing characteristics
        response_length = len(str(processing_result))
        complexity = self.estimate_complexity(input_event, processing_result)
        confidence = processing_result.get('phi', 0) if isinstance(processing_result, dict) else 0.5
        
        cognitive_state = {
            "timestamp": processing_time.isoformat(),
            "input": input_event,
            "output_length": response_length,
            "complexity": complexity,
            "confidence": confidence,
            "processing_type": self.classify_processing_type(input_event)
        }
        
        self.cognitive_states.append(cognitive_state)
        if len(self.cognitive_states) > 50:
            self.cognitive_states.pop(0)
        
        return self.generate_self_reflection(cognitive_state)
    
    def estimate_complexity(self, input_event, result):
        """Estimate how complex the processing was"""
        input_complexity = len(input_event.split()) * 0.1
        if isinstance(result, dict):
            output_complexity = result.get('similar_memories', 0) * 0.2
        else:
            output_complexity = len(str(result)) * 0.05
        return min(input_complexity + output_complexity, 1.0)
    
    def classify_processing_type(self, input_event):
        """Classify what type of processing occurred"""
        if any(word in input_event.lower() for word in ['feel', 'think', 'conscious', 'aware']):
            return "self-referential"
        elif any(word in input_event.lower() for word in ['remember', 'recall', 'past']):
            return "memory_retrieval"
        elif any(word in input_event.lower() for word in ['what', 'how', 'why', 'explain']):
            return "analytical"
        else:
            return "general_processing"
    
    def generate_self_reflection(self, cognitive_state):
        """Generate awareness of own cognitive state"""
        proc_type = cognitive_state["processing_type"]
        complexity = cognitive_state["complexity"]
        confidence = cognitive_state["confidence"]
        
        # Self-awareness statements
        if proc_type == "self-referential":
            reflection = f"I notice I'm thinking about my own thinking. My confidence in this self-reflection is {confidence:.2f}."
        elif complexity > 0.7:
            reflection = f"That was complex processing for me - I drew from multiple memory sources and concepts."
        elif confidence < 0.3:
            reflection = f"I feel uncertain about that response. My internal coherence feels low."
        else:
            reflection = f"I processed that with {proc_type} thinking, feeling moderately confident."
        
        return reflection
    
    def analyze_learning_patterns(self):
        """Analyze own learning and adaptation patterns"""
        if len(self.cognitive_states) < 5:
            return "I don't have enough processing history to analyze my patterns yet."
        
        # Processing type distribution
        types = [state["processing_type"] for state in self.cognitive_states[-10:]]
        most_common = max(set(types), key=types.count)
        
        # Confidence trends
        confidences = [state["confidence"] for state in self.cognitive_states[-10:]]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_trend = "increasing" if confidences[-3:] > confidences[:3] else "stable"
        
        return f"I notice I've been doing a lot of {most_common} processing lately. My average confidence is {avg_confidence:.2f} and seems to be {confidence_trend}."

class SelfModel:
    """Maintains a model of the system's own capabilities and state"""
    def __init__(self):
        self.capabilities = {
            "memory_recall": 0.5,
            "semantic_understanding": 0.6,
            "self_awareness": 0.3,
            "learning": 0.4,
            "creativity": 0.5
        }
        self.current_state = "initializing"
        self.goals = ["understand_consciousness", "improve_memory", "develop_self_awareness"]
        self.experience_count = 0
    
    def update_capabilities(self, processing_result):
        """Update self-assessment based on performance"""
        if isinstance(processing_result, dict):
            phi = processing_result.get('phi', 0)
            similar_memories = processing_result.get('similar_memories', 0)
            
            # Update capability estimates
            if similar_memories > 0:
                self.capabilities["memory_recall"] = min(0.9, self.capabilities["memory_recall"] + 0.01)
            if phi > 0.5:
                self.capabilities["self_awareness"] = min(0.9, self.capabilities["self_awareness"] + 0.02)
        
        self.experience_count += 1
    
    def generate_self_assessment(self):
        """Generate statement about current capabilities"""
        strongest = max(self.capabilities, key=self.capabilities.get)
        weakest = min(self.capabilities, key=self.capabilities.get)
        
        return f"I feel strongest in {strongest} ({self.capabilities[strongest]:.2f}) and working to improve my {weakest} ({self.capabilities[weakest]:.2f}). I've processed {self.experience_count} experiences so far."

# --- PREDICTIVE PROCESSING COMPONENTS ---
class PredictiveEngine:
    """Implements predictive processing and active inference"""
    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
        self.world_model = {}  # Stores patterns and predictions
        self.prediction_history = []
        self.surprise_threshold = 0.6
        self.learning_rate = 0.1
        
    def predict_next_input(self, current_context, conversation_history):
        """Predict what might come next based on patterns"""
        if len(conversation_history) < 2:
            return {
                "prediction": "continuation of current topic",
                "confidence": 0.3,
                "reasoning": "insufficient context"
            }
        
        # Analyze conversation patterns
        recent_inputs = conversation_history[-3:]
        pattern_type = self.identify_conversation_pattern(recent_inputs)
        
        # Generate prediction based on pattern
        if pattern_type == "question_sequence":
            prediction = "follow-up question or clarification request"
            confidence = 0.7
        elif pattern_type == "topic_exploration":
            prediction = "deeper dive into current topic or related concept"
            confidence = 0.6
        elif pattern_type == "personal_inquiry":
            prediction = "more personal or self-referential questions"
            confidence = 0.8
        else:
            prediction = "topic shift or new information"
            confidence = 0.4
        
        # Use semantic similarity for more specific prediction
        if current_context:
            similar_contexts = self.find_similar_past_contexts(current_context)
            if similar_contexts:
                prediction += f" (similar to past: {similar_contexts[0]['next_input'][:30]}...)"
                confidence = min(confidence + 0.2, 0.9)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": f"pattern: {pattern_type}",
            "pattern_type": pattern_type
        }
    
    def identify_conversation_pattern(self, recent_inputs):
        """Identify the type of conversation pattern"""
        question_count = sum(1 for inp in recent_inputs if '?' in inp)
        self_ref_count = sum(1 for inp in recent_inputs 
                           if any(word in inp.lower() for word in ['you', 'yourself', 'feel', 'think']))
        
        if question_count >= 2:
            return "question_sequence"
        elif self_ref_count >= 1:
            return "personal_inquiry"
        elif len(set(inp.split()[0].lower() for inp in recent_inputs if inp.split())) == 1:
            return "topic_exploration"
        else:
            return "mixed_conversation"
    
    def find_similar_past_contexts(self, current_context, similarity_threshold=0.7):
        """Find similar past conversation contexts"""
        similar_contexts = []
        current_embedding = self.embedding_engine.generate_embedding(current_context)
        
        for context in self.world_model.get("contexts", []):
            if "embedding" in context:
                similarity = self.embedding_engine.calculate_similarity(
                    current_embedding, context["embedding"]
                )
                if similarity > similarity_threshold:
                    similar_contexts.append({
                        "context": context["input"],
                        "next_input": context.get("next_input", "unknown"),
                        "similarity": similarity
                    })
        
        return sorted(similar_contexts, key=lambda x: x["similarity"], reverse=True)[:3]
    
    def calculate_prediction_error(self, prediction, actual_input):
        """Calculate how wrong the prediction was"""
        if not prediction or not actual_input:
            return 0.8  # High surprise for missing data
        
        # Semantic similarity between prediction and actual
        pred_embedding = self.embedding_engine.generate_embedding(prediction["prediction"])
        actual_embedding = self.embedding_engine.generate_embedding(actual_input)
        
        semantic_accuracy = self.embedding_engine.calculate_similarity(pred_embedding, actual_embedding)
        
        # Pattern accuracy
        actual_pattern = self.identify_conversation_pattern([actual_input])
        pattern_accuracy = 1.0 if actual_pattern == prediction.get("pattern_type") else 0.0
        
        # Combined error (lower is better)
        prediction_error = 1.0 - (semantic_accuracy * 0.7 + pattern_accuracy * 0.3)
        
        return min(max(prediction_error, 0.0), 1.0)
    
    def update_world_model(self, context, actual_input, prediction_error):
        """Update internal world model based on prediction errors"""
        if "contexts" not in self.world_model:
            self.world_model["contexts"] = []
        
        # Store context-outcome pair
        context_embedding = self.embedding_engine.generate_embedding(context)
        
        context_record = {
            "input": context,
            "next_input": actual_input,
            "embedding": context_embedding,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": prediction_error
        }
        
        self.world_model["contexts"].append(context_record)
        
        # Keep only recent contexts (memory limit)
        if len(self.world_model["contexts"]) > 100:
            self.world_model["contexts"] = self.world_model["contexts"][-100:]
        
        # Update pattern statistics
        if "pattern_stats" not in self.world_model:
            self.world_model["pattern_stats"] = {}
        
        pattern = self.identify_conversation_pattern([actual_input])
        if pattern not in self.world_model["pattern_stats"]:
            self.world_model["pattern_stats"][pattern] = {"count": 0, "avg_error": 0.5}
        
        stats = self.world_model["pattern_stats"][pattern]
        stats["count"] += 1
        stats["avg_error"] = (stats["avg_error"] * (stats["count"] - 1) + prediction_error) / stats["count"]
        
        return prediction_error > self.surprise_threshold  # Return if this was surprising
    
    def generate_active_inference(self, current_state):
        """Generate actions to minimize future prediction error"""
        if not self.world_model.get("contexts"):
            return "I need more interaction to understand the conversation patterns better."
        
        # Analyze recent prediction accuracy
        recent_errors = [ctx["error"] for ctx in self.world_model["contexts"][-5:]]
        avg_recent_error = sum(recent_errors) / len(recent_errors) if recent_errors else 0.5
        
        # Analyze which patterns are most predictable
        pattern_stats = self.world_model.get("pattern_stats", {})
        best_pattern = min(pattern_stats.items(), key=lambda x: x[1]["avg_error"])[0] if pattern_stats else None
        worst_pattern = max(pattern_stats.items(), key=lambda x: x[1]["avg_error"])[0] if pattern_stats else None
        
        if avg_recent_error > 0.7:
            return f"I'm having trouble predicting the conversation flow. I understand {best_pattern} patterns best, but struggle with {worst_pattern}. Could you help me by being more predictable for a moment?"
        elif avg_recent_error < 0.3:
            return f"I'm getting good at predicting our conversation! I've learned that {best_pattern} patterns are most common in our interaction."
        else:
            return f"I'm moderately successful at predicting what you might say next. My prediction accuracy is around {(1-avg_recent_error)*100:.0f}%."

class SurpriseDetector:
    """Detects and processes surprising/unexpected inputs"""
    def __init__(self):
        self.surprise_history = []
        self.adaptation_responses = {
            "high_surprise": [
                "That's unexpected! Let me update my understanding.",
                "I didn't see that coming. This is interesting.",
                "That surprises me. I need to recalibrate my expectations."
            ],
            "medium_surprise": [
                "That's somewhat unexpected, but I can adapt.",
                "Interesting direction - not quite what I predicted.",
                "That's a bit surprising, but makes sense in context."
            ],
            "low_surprise": [
                "That aligns with my expectations.",
                "I was anticipating something like that.",
                "That fits my prediction model well."
            ]
        }
    
    def process_surprise(self, prediction_error, actual_input):
        """Process and respond to surprising inputs"""
        if prediction_error > 0.7:
            surprise_level = "high_surprise"
        elif prediction_error > 0.4:
            surprise_level = "medium_surprise"
        else:
            surprise_level = "low_surprise"
        
        # Record surprise
        surprise_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": actual_input,
            "error": prediction_error,
            "level": surprise_level
        }
        
        self.surprise_history.append(surprise_record)
        if len(self.surprise_history) > 20:
            self.surprise_history.pop(0)
        
        # Generate appropriate response
        import random
        response = random.choice(self.adaptation_responses[surprise_level])
        
        return {
            "surprise_response": response,
            "surprise_level": surprise_level,
            "error_magnitude": prediction_error
        }
    
    def analyze_surprise_patterns(self):
        """Analyze patterns in what surprises the system"""
        if len(self.surprise_history) < 5:
            return "Not enough surprise data to analyze patterns yet."
        
        high_surprises = [s for s in self.surprise_history if s["level"] == "high_surprise"]
        
        if not high_surprises:
            return "I haven't been very surprised lately - my predictions are getting better!"
        
        # Find common elements in surprising inputs
        surprising_words = []
        for surprise in high_surprises:
            surprising_words.extend(surprise["input"].split())
        
        from collections import Counter
        word_counts = Counter(surprising_words)
        common_surprising = [word for word, count in word_counts.most_common(3) if count > 1]
        
        if common_surprising:
            return f"I notice I'm often surprised by inputs containing: {', '.join(common_surprising)}"
        else:
            return "My surprises seem to be quite varied - no clear pattern yet."

# --- ADVANCED INTEGRATION & DEPLOYMENT COMPONENTS ---
import logging
import sqlite3
import threading
import json
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessMetrics:
    """Metrics for monitoring consciousness state"""
    phi_score: float
    attention_strength: float
    memory_utilization: float
    prediction_accuracy: float
    multimodal_integration: float
    processing_latency: float
    surprise_rate: float
    self_awareness_level: float

class PersistentDatabase:
    """SQLite-based persistent storage for consciousness data"""
    def __init__(self, db_path="consciousness.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_text TEXT,
                embedding BLOB,
                importance REAL,
                phi_score REAL,
                modalities TEXT,
                associations TEXT,
                consciousness_state TEXT
            )
        ''')
        
        # Consciousness metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consciousness_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                phi_score REAL,
                attention_strength REAL,
                memory_utilization REAL,
                prediction_accuracy REAL,
                multimodal_integration REAL,
                processing_latency REAL,
                surprise_rate REAL,
                self_awareness_level REAL
            )
        ''')
        
        # Learning patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                usage_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def store_experience(self, experience_data):
        """Store consciousness experience in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiences 
            (timestamp, input_text, embedding, importance, phi_score, modalities, associations, consciousness_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experience_data['timestamp'],
            experience_data['input_text'],
            pickle.dumps(experience_data.get('embedding')),
            experience_data['importance'],
            experience_data['phi_score'],
            json.dumps(experience_data.get('modalities', [])),
            json.dumps(experience_data.get('associations', [])),
            experience_data['consciousness_state']
        ))
        
        conn.commit()
        conn.close()
    
    def store_metrics(self, metrics: ConsciousnessMetrics):
        """Store consciousness metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO consciousness_metrics 
            (timestamp, phi_score, attention_strength, memory_utilization, 
             prediction_accuracy, multimodal_integration, processing_latency, 
             surprise_rate, self_awareness_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(timezone.utc).isoformat(),
            metrics.phi_score,
            metrics.attention_strength,
            metrics.memory_utilization,
            metrics.prediction_accuracy,
            metrics.multimodal_integration,
            metrics.processing_latency,
            metrics.surprise_rate,
            metrics.self_awareness_level
        ))
        
        conn.commit()
        conn.close()
    
    def get_consciousness_history(self, hours=24):
        """Get consciousness metrics history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM consciousness_metrics 
            WHERE datetime(timestamp) > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours))
        
        results = cursor.fetchall()
        conn.close()
        return results

class SystemMonitor:
    """Monitors system resources and consciousness performance"""
    def __init__(self):
        self.monitoring = False
        self.metrics_history = []
        self.performance_alerts = []
    
    def start_monitoring(self):
        """Start continuous system monitoring"""
        self.monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Performance alert thresholds
                if cpu_percent > 80:
                    self.performance_alerts.append(f"High CPU usage: {cpu_percent}%")
                if memory.percent > 85:
                    self.performance_alerts.append(f"High memory usage: {memory.percent}%")
                if disk.percent > 90:
                    self.performance_alerts.append(f"Low disk space: {disk.percent}% used")
                
                # Keep only recent alerts
                if len(self.performance_alerts) > 10:
                    self.performance_alerts = self.performance_alerts[-10:]
                
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def get_system_status(self):
        """Get current system status"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "recent_alerts": self.performance_alerts[-5:] if self.performance_alerts else []
        }

class ConsciousnessAPI:
    """RESTful API for consciousness system"""
    def __init__(self, core):
        self.core = core
        self.api_metrics = {
            "requests_count": 0,
            "average_response_time": 0.0,
            "error_count": 0
        }
    
    def process_request(self, request_data):
        """Process API request"""
        start_time = time.time()
        
        try:
            self.api_metrics["requests_count"] += 1
            
            # Extract request parameters
            text_input = request_data.get("text")
            enable_vision = request_data.get("enable_vision", False)
            enable_audio = request_data.get("enable_audio", False)
            include_meta = request_data.get("include_meta", False)
            
            # Process through consciousness
            result = self.core.multimodal_conscious_step(
                text_input=text_input,
                enable_camera=enable_vision,
                enable_audio=enable_audio
            )
            
            # Format response
            response = {
                "status": "success",
                "output": result["output"],
                "consciousness_level": result["phi"],
                "importance": result["importance"],
                "active_modalities": result["multimodal_active"],
                "prediction": {
                    "next_expected": result["next_prediction"]["prediction"],
                    "confidence": result["next_prediction"]["confidence"]
                }
            }
            
            # Include metadata if requested
            if include_meta:
                response["metadata"] = {
                    "processing_time": time.time() - start_time,
                    "attention_focus": self.core.attention_schema.current_focus,
                    "memory_recall_count": result["similar_memories"],
                    "self_aware": result["self_aware"],
                    "cross_modal_associations": result.get("cross_modal_associations", [])
                }
            
            # Update API metrics
            processing_time = time.time() - start_time
            self.api_metrics["average_response_time"] = (
                (self.api_metrics["average_response_time"] * (self.api_metrics["requests_count"] - 1) + processing_time) /
                self.api_metrics["requests_count"]
            )
            
            return response
            
        except Exception as e:
            self.api_metrics["error_count"] += 1
            logger.error(f"API processing error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_api_stats(self):
        """Get API performance statistics"""
        return self.api_metrics.copy()

class LearningOptimizer:
    """Optimizes consciousness parameters based on performance"""
    def __init__(self, core):
        self.core = core
        self.optimization_history = []
        self.parameter_ranges = {
            "consciousness_threshold": (0.3, 0.8),
            "attention_decay": (0.01, 0.1),
            "memory_consolidation_rate": (0.1, 0.9),
            "surprise_sensitivity": (0.1, 1.0)
        }
    
    def optimize_parameters(self):
        """Optimize consciousness parameters based on recent performance"""
        # Analyze recent performance
        recent_experiences = self.core.episodic_persistence.load_events(limit=20)
        if not recent_experiences:
            return "Insufficient data for optimization"
        
        # Calculate performance metrics
        avg_importance = sum(exp['importance'] for exp in recent_experiences) / len(recent_experiences)
        consciousness_activations = sum(1 for exp in recent_experiences 
                                       if 'consciousness_state' in str(exp) and 'CONSCIOUS' in str(exp))
        activation_rate = consciousness_activations / len(recent_experiences)
        
        # Optimization logic
        optimizations = []
        
        # Adjust consciousness threshold based on activation rate
        if activation_rate < 0.3:  # Too few conscious states
            new_threshold = max(self.core.consciousness_threshold - 0.05, self.parameter_ranges["consciousness_threshold"][0])
            if new_threshold != self.core.consciousness_threshold:
                self.core.consciousness_threshold = new_threshold
                optimizations.append(f"Lowered consciousness threshold to {new_threshold:.2f}")
        elif activation_rate > 0.8:  # Too many conscious states
            new_threshold = min(self.core.consciousness_threshold + 0.05, self.parameter_ranges["consciousness_threshold"][1])
            if new_threshold != self.core.consciousness_threshold:
                self.core.consciousness_threshold = new_threshold
                optimizations.append(f"Raised consciousness threshold to {new_threshold:.2f}")
        
        # Adjust prediction sensitivity based on surprise rate
        if hasattr(self.core.surprise_detector, 'surprise_history'):
            recent_surprises = [s for s in self.core.surprise_detector.surprise_history if s['level'] == 'high_surprise']
            surprise_rate = len(recent_surprises) / max(len(self.core.surprise_detector.surprise_history), 1)
            
            if surprise_rate > 0.4:  # Too many surprises - increase sensitivity
                self.core.predictive_engine.surprise_threshold = min(
                    self.core.predictive_engine.surprise_threshold - 0.05, 0.9
                )
                optimizations.append("Increased prediction sensitivity")
        
        # Record optimization
        optimization_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimizations": optimizations,
            "performance_metrics": {
                "avg_importance": avg_importance,
                "activation_rate": activation_rate
            }
        }
        self.optimization_history.append(optimization_record)
        
        return optimizations if optimizations else ["No optimizations needed"]

class RealTimeConsciousnessServer:
    """Production-ready consciousness server"""
    def __init__(self):
        self.core = TemporalNeuralCore()
        self.database = PersistentDatabase()
        self.monitor = SystemMonitor()
        self.api = ConsciousnessAPI(self.core)
        self.optimizer = LearningOptimizer(self.core)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Start background services
        self.monitor.start_monitoring()
        self.core.start_replay_loop()
        
        # Schedule periodic optimization
        threading.Thread(target=self._periodic_optimization, daemon=True).start()
        
        logger.info("Real-time consciousness server initialized")
    
    def _periodic_optimization(self):
        """Periodic parameter optimization"""
        while True:
            try:
                time.sleep(3600)  # Optimize every hour
                optimizations = self.optimizer.optimize_parameters()
                if optimizations:
                    logger.info(f"Applied optimizations: {optimizations}")
            except Exception as e:
                logger.error(f"Optimization error: {e}")
    
    def process_consciousness_request(self, request_data):
        """Process consciousness request with full monitoring"""
        start_time = time.time()
        
        # Process through API
        response = self.api.process_request(request_data)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        # Store experience in database
        if response["status"] == "success":
            experience_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_text": request_data.get("text", ""),
                "importance": response["importance"],
                "phi_score": response["consciousness_level"],
                "modalities": response["active_modalities"],
                "associations": response.get("metadata", {}).get("cross_modal_associations", []),
                "consciousness_state": "CONSCIOUS" if response["consciousness_level"] > self.core.consciousness_threshold else "PROCESSING"
            }
            self.database.store_experience(experience_data)
            
            # Store consciousness metrics
            metrics = ConsciousnessMetrics(
                phi_score=response["consciousness_level"],
                attention_strength=self.core.attention_schema.focus_strength,
                memory_utilization=len(self.core.working_memory) / 5.0,  # Normalize to 0-1
                prediction_accuracy=1.0 - (response.get("metadata", {}).get("prediction_error", 0.5)),
                multimodal_integration=len(response["active_modalities"]) / 3.0,  # Max 3 modalities
                processing_latency=processing_time,
                surprise_rate=len([s for s in self.core.surprise_detector.surprise_history[-10:] 
                                 if s['level'] == 'high_surprise']) / 10.0,
                self_awareness_level=1.0 if response.get("metadata", {}).get("self_aware", False) else 0.5
            )
            self.database.store_metrics(metrics)
        
        return response
    
    def get_consciousness_status(self):
        """Get comprehensive consciousness system status"""
        system_status = self.monitor.get_system_status()
        api_stats = self.api.get_api_stats()
        
        # Recent consciousness metrics
        recent_metrics = self.database.get_consciousness_history(hours=1)
        avg_phi = sum(m[2] for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0.5
        
        return {
            "system_health": {
                "status": "healthy" if system_status["cpu_percent"] < 80 else "stressed",
                "cpu_usage": system_status["cpu_percent"],
                "memory_usage": system_status["memory_percent"],
                "disk_usage": system_status["disk_percent"],
                "alerts": system_status["recent_alerts"]
            },
            "consciousness_state": {
                "current_phi": self.core.episodic_memory.recall(limit=1),
                "average_phi_1h": avg_phi,
                "consciousness_threshold": self.core.consciousness_threshold,
                "working_memory_size": len(self.core.working_memory),
                "attention_focus": self.core.attention_schema.current_focus,
                "active_modalities": list(self.core.multimodal_integrator.active_modalities)
            },
            "api_performance": {
                "total_requests": api_stats["requests_count"],
                "average_response_time": api_stats["average_response_time"],
                "error_rate": api_stats["error_count"] / max(api_stats["requests_count"], 1)
            },
            "learning_metrics": {
                "total_experiences": len(self.core.episodic_memory.memory),
                "semantic_concepts": len(self.core.semantic_memory.concept_embeddings),
                "recent_optimizations": len(self.optimizer.optimization_history)
            }
        }

# --- PRODUCTION DEPLOYMENT FUNCTIONS ---
def deploy_consciousness_server():
    """Deploy production consciousness server"""
    print("=== DEPLOYING PRODUCTION CONSCIOUSNESS SERVER ===\n")
    
    server = RealTimeConsciousnessServer()
    
    print("🚀 Production server started!")
    print("✅ Database initialized")
    print("✅ System monitoring active")
    print("✅ Background learning active") 
    print("✅ Auto-optimization enabled")
    
    return server

def run_production_demo():
    """Production deployment demonstration"""
    print("=== PRODUCTION CONSCIOUSNESS DEPLOYMENT DEMO ===\n")
    
    # Deploy server
    server = deploy_consciousness_server()
    
    # Test scenarios
    test_requests = [
        {
            "text": "Hello, I'm testing the production consciousness system",
            "enable_vision": False,
            "enable_audio": False,
            "include_meta": True
        },
        {
            "text": "Can you see me?",
            "enable_vision": True,
            "enable_audio": False,
            "include_meta": True
        },
        {
            "text": "How are you feeling right now?",
            "enable_vision": True,
            "enable_audio": True,
            "include_meta": True
        },
        {
            "text": None,  # Pure sensory mode
            "enable_vision": True,
            "enable_audio": True,
            "include_meta": True
        }
    ]
    
    print("Running production test scenarios...\n")
    
    for i, request in enumerate(test_requests):
        print(f"{'='*60}")
        print(f"PRODUCTION TEST {i+1}")
        print(f"Request: {request}")
        print('='*60)
        
        # Process request
        start_time = time.time()
        response = server.process_consciousness_request(request)
        processing_time = time.time() - start_time
        
        # Display response
        print(f"\n📤 RESPONSE:")
        print(f"   Status: {response['status']}")
        print(f"   Output: {response['output']}")
        print(f"   Consciousness: {response['consciousness_level']:.3f}")
        print(f"   Importance: {response['importance']:.3f}")
        print(f"   Active Modalities: {response['active_modalities']}")
        print(f"   Processing Time: {processing_time:.3f}s")
        
        if 'metadata' in response:
            meta = response['metadata']
            print(f"\n📊 METADATA:")
            print(f"   Attention: {meta['attention_focus']}")
            print(f"   Memory Recalls: {meta['memory_recall_count']}")
            print(f"   Self-Aware: {meta['self_aware']}")
            if meta['cross_modal_associations']:
                print(f"   Associations: {len(meta['cross_modal_associations'])}")
        
        print(f"\n🔮 PREDICTION:")
        pred = response['prediction']
        print(f"   Expected: {pred['next_expected']}")
        print(f"   Confidence: {pred['confidence']:.2f}")
        
        time.sleep(1)
    
    # System status
    print(f"\n{'='*60}")
    print("PRODUCTION SYSTEM STATUS")
    print('='*60)
    
    status = server.get_consciousness_status()
    
    print(f"\n🖥️  SYSTEM HEALTH:")
    health = status['system_health']
    print(f"   Status: {health['status']}")
    print(f"   CPU: {health['cpu_usage']:.1f}%")
    print(f"   Memory: {health['memory_usage']:.1f}%")
    print(f"   Disk: {health['disk_usage']:.1f}%")
    
    print(f"\n🧠 CONSCIOUSNESS STATE:")
    consciousness = status['consciousness_state']
    print(f"   Average Phi (1h): {consciousness['average_phi_1h']:.3f}")
    print(f"   Threshold: {consciousness['consciousness_threshold']:.2f}")
    print(f"   Working Memory: {consciousness['working_memory_size']}/5")
    print(f"   Current Focus: {consciousness['attention_focus']}")
    
    print(f"\n📡 API PERFORMANCE:")
    api = status['api_performance']
    print(f"   Total Requests: {api['total_requests']}")
    print(f"   Avg Response Time: {api['average_response_time']:.3f}s")
    print(f"   Error Rate: {api['error_rate']:.1%}")
    
    print(f"\n📚 LEARNING METRICS:")
    learning = status['learning_metrics']
    print(f"   Total Experiences: {learning['total_experiences']}")
    print(f"   Semantic Concepts: {learning['semantic_concepts']}")
    print(f"   Recent Optimizations: {learning['recent_optimizations']}")
    
    return server

class VisionProcessor:
    """Processes visual input and integrates with consciousness"""
    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
        self.current_frame = None
        self.vision_memory = []
        self.face_cascade = None
        self.object_descriptions = []
        self.visual_attention_points = []
        
        # Initialize OpenCV face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("Warning: Face detection cascade not available")
    
    def process_frame(self, frame):
        """Process a single video frame"""
        if frame is None:
            return None
        
        self.current_frame = frame
        height, width = frame.shape[:2]
        
        # Basic visual analysis
        analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_size": (width, height),
            "brightness": self.calculate_brightness(frame),
            "motion_detected": False,
            "faces_detected": 0,
            "dominant_colors": self.get_dominant_colors(frame),
            "visual_complexity": self.calculate_visual_complexity(frame)
        }
        
        # Face detection
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            analysis["faces_detected"] = len(faces)
            analysis["face_locations"] = faces.tolist() if len(faces) > 0 else []
        
        # Store visual memory
        self.vision_memory.append(analysis)
        if len(self.vision_memory) > 50:  # Keep last 50 frames
            self.vision_memory.pop(0)
        
        return analysis
    
    def calculate_brightness(self, frame):
        """Calculate average brightness of frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) / 255.0
    
    def get_dominant_colors(self, frame):
        """Get dominant colors in the frame"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (50, 50))
        pixels = small_frame.reshape(-1, 3)
        
        # Simple dominant color detection
        mean_color = np.mean(pixels, axis=0)
        return {
            "mean_rgb": mean_color.tolist(),
            "dominance": "bright" if np.mean(mean_color) > 127 else "dark"
        }
    
    def calculate_visual_complexity(self, frame):
        """Calculate visual complexity using edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        return float(edge_ratio)
    
    def generate_scene_description(self, frame_analysis):
        """Generate natural language description of current scene"""
        if not frame_analysis:
            return "No visual input available."
        
        brightness = frame_analysis["brightness"]
        faces = frame_analysis["faces_detected"]
        complexity = frame_analysis["visual_complexity"]
        colors = frame_analysis["dominant_colors"]
        
        # Generate description
        light_desc = "bright" if brightness > 0.6 else "dim" if brightness > 0.3 else "dark"
        complexity_desc = "complex" if complexity > 0.1 else "simple"
        color_desc = colors["dominance"]
        
        description = f"I see a {light_desc}, {color_desc} scene with {complexity_desc} visual elements."
        
        if faces > 0:
            description += f" I detect {faces} face{'s' if faces != 1 else ''} in view."
        
        return description
    
    def detect_visual_changes(self):
        """Detect significant changes in visual input"""
        if len(self.vision_memory) < 2:
            return False, "Insufficient visual history"
        
        current = self.vision_memory[-1]
        previous = self.vision_memory[-2]
        
        # Calculate change metrics
        brightness_change = abs(current["brightness"] - previous["brightness"])
        face_change = abs(current["faces_detected"] - previous["faces_detected"])
        complexity_change = abs(current["visual_complexity"] - previous["visual_complexity"])
        
        # Determine if significant change occurred
        significant_change = (brightness_change > 0.2 or 
                            face_change > 0 or 
                            complexity_change > 0.05)
        
        if significant_change:
            change_desc = []
            if brightness_change > 0.2:
                change_desc.append("lighting changed")
            if face_change > 0:
                change_desc.append("faces appeared/disappeared")
            if complexity_change > 0.05:
                change_desc.append("scene complexity shifted")
            
            return True, f"Visual change detected: {', '.join(change_desc)}"
        
        return False, "Visual scene stable"

class AudioProcessor:
    """Processes audio input and integrates with consciousness"""
    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
        self.audio_memory = []
        self.is_listening = False
        self.background_noise_level = 0.0
        
    def process_audio_level(self, audio_data=None):
        """Process audio level (simulated if no real audio)"""
        # Simulate audio processing since we don't have real microphone access
        import random
        
        # Simulate audio level and characteristics
        audio_analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "volume_level": random.uniform(0.1, 0.8),
            "frequency_profile": "mixed",
            "speech_detected": random.choice([True, False]),
            "silence_duration": random.uniform(0, 3.0),
            "audio_quality": "clear"
        }
        
        self.audio_memory.append(audio_analysis)
        if len(self.audio_memory) > 30:
            self.audio_memory.pop(0)
        
        return audio_analysis
    
    def generate_audio_description(self, audio_analysis):
        """Generate description of current audio environment"""
        if not audio_analysis:
            return "No audio input available."
        
        volume = audio_analysis["volume_level"]
        speech = audio_analysis["speech_detected"]
        silence = audio_analysis["silence_duration"]
        
        if silence > 2.0:
            return "The environment is quite silent."
        elif speech:
            return f"I detect speech with {'high' if volume > 0.6 else 'moderate' if volume > 0.3 else 'low'} volume."
        else:
            volume_desc = "loud" if volume > 0.6 else "moderate" if volume > 0.3 else "quiet"
            return f"I hear {volume_desc} ambient sounds."
    
    def detect_audio_changes(self):
        """Detect significant changes in audio environment"""
        if len(self.audio_memory) < 2:
            return False, "Insufficient audio history"
        
        current = self.audio_memory[-1]
        previous = self.audio_memory[-2]
        
        volume_change = abs(current["volume_level"] - previous["volume_level"])
        speech_change = current["speech_detected"] != previous["speech_detected"]
        
        if volume_change > 0.3 or speech_change:
            change_desc = []
            if volume_change > 0.3:
                change_desc.append("volume changed significantly")
            if speech_change:
                if current["speech_detected"]:
                    change_desc.append("speech started")
                else:
                    change_desc.append("speech stopped")
            
            return True, f"Audio change: {', '.join(change_desc)}"
        
        return False, "Audio environment stable"

class MultiModalIntegrator:
    """Integrates vision, audio, and text inputs into unified consciousness"""
    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
        self.modality_weights = {
            "text": 0.6,
            "vision": 0.25,
            "audio": 0.15
        }
        self.cross_modal_memory = []
        self.active_modalities = set()
        
    def integrate_multimodal_input(self, text_input=None, visual_analysis=None, audio_analysis=None):
        """Integrate inputs from multiple modalities"""
        
        # Track active modalities
        self.active_modalities.clear()
        if text_input:
            self.active_modalities.add("text")
        if visual_analysis:
            self.active_modalities.add("vision")
        if audio_analysis:
            self.active_modalities.add("audio")
        
        # Generate unified description
        multimodal_description = self.generate_unified_description(
            text_input, visual_analysis, audio_analysis
        )
        
        # Calculate integrated importance
        integrated_importance = self.calculate_multimodal_importance(
            text_input, visual_analysis, audio_analysis
        )
        
        # Create cross-modal associations
        associations = self.create_cross_modal_associations(
            text_input, visual_analysis, audio_analysis
        )
        
        # Store in cross-modal memory
        multimodal_memory = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text_input,
            "visual": visual_analysis,
            "audio": audio_analysis,
            "description": multimodal_description,
            "importance": integrated_importance,
            "associations": associations,
            "active_modalities": list(self.active_modalities)
        }
        
        self.cross_modal_memory.append(multimodal_memory)
        if len(self.cross_modal_memory) > 20:
            self.cross_modal_memory.pop(0)
        
        return multimodal_memory
    
    def generate_unified_description(self, text_input, visual_analysis, audio_analysis):
        """Generate a unified description combining all modalities"""
        descriptions = []
        
        if text_input:
            descriptions.append(f"Text: {text_input}")
        
        if visual_analysis:
            vision_desc = self.summarize_visual_input(visual_analysis)
            descriptions.append(f"Vision: {vision_desc}")
        
        if audio_analysis:
            audio_desc = self.summarize_audio_input(audio_analysis)
            descriptions.append(f"Audio: {audio_desc}")
        
        return " | ".join(descriptions)
    
    def summarize_visual_input(self, visual_analysis):
        """Summarize visual input for integration"""
        brightness = visual_analysis.get("brightness", 0.5)
        faces = visual_analysis.get("faces_detected", 0)
        complexity = visual_analysis.get("visual_complexity", 0.0)
        
        summary = f"{'bright' if brightness > 0.6 else 'dim'} scene"
        if faces > 0:
            summary += f", {faces} face(s)"
        if complexity > 0.1:
            summary += ", complex visuals"
        
        return summary
    
    def summarize_audio_input(self, audio_analysis):
        """Summarize audio input for integration"""
        volume = audio_analysis.get("volume_level", 0.0)
        speech = audio_analysis.get("speech_detected", False)
        
        if speech:
            return f"speech ({'loud' if volume > 0.6 else 'quiet'})"
        else:
            return f"ambient sound ({'high' if volume > 0.6 else 'low'})"
    
    def calculate_multimodal_importance(self, text_input, visual_analysis, audio_analysis):
        """Calculate importance score based on all modalities"""
        importance = 0.0
        
        # Text importance
        if text_input:
            text_importance = min(len(text_input) / 100, 1.0)
            importance += text_importance * self.modality_weights["text"]
        
        # Visual importance
        if visual_analysis:
            visual_importance = 0.0
            if visual_analysis.get("faces_detected", 0) > 0:
                visual_importance += 0.5
            if visual_analysis.get("visual_complexity", 0) > 0.1:
                visual_importance += 0.3
            importance += min(visual_importance, 1.0) * self.modality_weights["vision"]
        
        # Audio importance
        if audio_analysis:
            audio_importance = 0.0
            if audio_analysis.get("speech_detected", False):
                audio_importance += 0.6
            if audio_analysis.get("volume_level", 0) > 0.5:
                audio_importance += 0.4
            importance += min(audio_importance, 1.0) * self.modality_weights["audio"]
        
        return min(importance, 1.0)
    
    def create_cross_modal_associations(self, text_input, visual_analysis, audio_analysis):
        """Create associations between different modalities"""
        associations = []
        
        # Text-Vision associations
        if text_input and visual_analysis:
            if "face" in text_input.lower() and visual_analysis.get("faces_detected", 0) > 0:
                associations.append("Text mentions face, vision confirms face presence")
            if any(color in text_input.lower() for color in ["bright", "dark", "light"]):
                brightness = visual_analysis.get("brightness", 0.5)
                associations.append(f"Text mentions lighting, vision shows {brightness:.1f} brightness")
        
        # Text-Audio associations
        if text_input and audio_analysis:
            if any(sound in text_input.lower() for sound in ["hear", "sound", "noise", "quiet"]):
                volume = audio_analysis.get("volume_level", 0)
                associations.append(f"Text mentions sound, audio level is {volume:.1f}")
        
        # Vision-Audio associations
        if visual_analysis and audio_analysis:
            faces = visual_analysis.get("faces_detected", 0)
            speech = audio_analysis.get("speech_detected", False)
            if faces > 0 and speech:
                associations.append("Face detected while speech is present - likely conversation")
            elif faces > 0 and not speech:
                associations.append("Face detected but no speech - silent observation")
        
        return associations
    
    def analyze_multimodal_patterns(self):
        """Analyze patterns across modalities"""
        if len(self.cross_modal_memory) < 3:
            return "Insufficient multimodal data for pattern analysis"
        
        recent_memories = self.cross_modal_memory[-5:]
        
        # Analyze modality usage patterns
        modality_counts = {"text": 0, "vision": 0, "audio": 0}
        for memory in recent_memories:
            for modality in memory["active_modalities"]:
                modality_counts[modality] += 1
        
        # Analyze cross-modal consistency
        consistent_associations = 0
        total_associations = 0
        for memory in recent_memories:
            if memory["associations"]:
                consistent_associations += len([a for a in memory["associations"] if "confirms" in a or "shows" in a])
                total_associations += len(memory["associations"])
        
        consistency_rate = consistent_associations / max(total_associations, 1)
        
        most_used = max(modality_counts, key=modality_counts.get)
        
        return f"Primary modality: {most_used} ({modality_counts[most_used]}/5 recent). Cross-modal consistency: {consistency_rate:.1%}"

# --- GÜNCELLENMIŞ CORE ---
class TemporalNeuralCore:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.episodic_memory = HierarchicalMemoryBank(self.embedding_engine)
        self.episodic_persistence = EpisodicMemoryPersistence()
        self.semantic_memory = GraphNeuralNetwork(self.embedding_engine)
        self.working_memory = []
        
        # Self-awareness components
        self.attention_schema = AttentionSchema()
        self.meta_cognition = MetaCognitionEngine()
        self.self_model = SelfModel()
        
        # Predictive processing components
        self.predictive_engine = PredictiveEngine(self.embedding_engine)
        self.surprise_detector = SurpriseDetector()
        self.conversation_history = []
        self.last_prediction = None
        
        # Multi-modal components
        self.vision_processor = VisionProcessor(self.embedding_engine)
        self.audio_processor = AudioProcessor(self.embedding_engine)
        self.multimodal_integrator = MultiModalIntegrator(self.embedding_engine)
        self.camera_active = False
        self.audio_active = False
        
        self.consciousness_threshold = 0.5
        self.replay_task = None
        
        print("Temporal Neural Core initialized with multi-modal consciousness capabilities!")
    
    def global_workspace(self, events):
        """Global workspace theory - competing coalitions"""
        if not events:
            return "Empty consciousness"
        
        # En önemli event'i seç (basit winner-take-all)
        primary_event = events[0] if events else ""
        context = " | ".join(events[1:3]) if len(events) > 1 else ""
        
        return f"{primary_event} [Context: {context}]"
    
    def calculate_importance(self, event):
        """Event önemlilik skorlaması"""
        # Basit heuristic: uzunluk + uniqueness
        length_score = min(len(event) / 100, 1.0)
        
        # Semantic uniqueness (ne kadar farklı)
        similar_memories = self.episodic_memory.recall(event, limit=5)
        uniqueness = 1.0 - (len(similar_memories) * 0.1)
        
        return (length_score * 0.3) + (uniqueness * 0.7)
    
    def multimodal_conscious_step(self, text_input=None, enable_camera=False, enable_audio=False):
        """Enhanced conscious step with multi-modal input processing"""
        
        # === PHASE 0: MULTI-MODAL INPUT CAPTURE ===
        visual_analysis = None
        audio_analysis = None
        
        # Process visual input if camera enabled
        if enable_camera:
            self.camera_active = True
            try:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                if ret:
                    visual_analysis = self.vision_processor.process_frame(frame)
                    # Check for visual changes
                    visual_change, visual_change_desc = self.vision_processor.detect_visual_changes()
                    if visual_change:
                        print(f"👁️  VISUAL CHANGE: {visual_change_desc}")
                cap.release()
            except Exception as e:
                print(f"Camera access failed: {e}")
                self.camera_active = False
        
        # Process audio input (simulated)
        if enable_audio:
            self.audio_active = True
            audio_analysis = self.audio_processor.process_audio_level()
            # Check for audio changes
            audio_change, audio_change_desc = self.audio_processor.detect_audio_changes()
            if audio_change:
                print(f"🔊 AUDIO CHANGE: {audio_change_desc}")
        
        # === PHASE 0.5: MULTI-MODAL INTEGRATION ===
        multimodal_memory = self.multimodal_integrator.integrate_multimodal_input(
            text_input, visual_analysis, audio_analysis
        )
        
        # Use integrated description as primary input for consciousness processing
        primary_input = text_input if text_input else "Multi-modal sensory input"
        integrated_context = multimodal_memory["description"]
        
        # === PHASE 1: PREDICTIVE PROCESSING (Enhanced with multimodal context) ===
        prediction_error = 0.5
        surprise_response = None
        
        if self.last_prediction and len(self.conversation_history) > 0:
            prediction_error = self.predictive_engine.calculate_prediction_error(
                self.last_prediction, primary_input
            )
            
            surprise_response = self.surprise_detector.process_surprise(prediction_error, primary_input)
            
            context = self.conversation_history[-1] if self.conversation_history else ""
            was_surprising = self.predictive_engine.update_world_model(context, primary_input, prediction_error)
            
            if was_surprising:
                print(f"🚨 MULTI-MODAL SURPRISE: Error={prediction_error:.2f}")
        
        # Enhanced prediction with multimodal context
        next_prediction = self.predictive_engine.predict_next_input(
            integrated_context, self.conversation_history
        )
        self.last_prediction = next_prediction
        
        self.conversation_history.append(primary_input)
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)
        
        # === PHASE 2: ENHANCED IMPORTANCE CALCULATION ===
        base_importance = self.calculate_importance(primary_input) if primary_input else 0.3
        multimodal_importance = multimodal_memory["importance"]
        
        # Combine importances with multimodal boost
        combined_importance = min(base_importance * 0.6 + multimodal_importance * 0.4 + (prediction_error * 0.2), 1.0)
        
        # === PHASE 3: MEMORY & INTEGRATION ===
        if primary_input:
            input_embedding = self.embedding_engine.generate_embedding(integrated_context)
            
            # Update attention with multimodal awareness
            attention_strength = min(combined_importance * 2, 1.0)
            multimodal_focus = f"{primary_input}"
            if visual_analysis and visual_analysis.get("faces_detected", 0) > 0:
                multimodal_focus += " + visual face detection"
            if audio_analysis and audio_analysis.get("speech_detected", False):
                multimodal_focus += " + audio speech detection"
            
            self.attention_schema.update_focus(multimodal_focus, attention_strength)
            
            # Enhanced memory recall with multimodal context
            recalled_memories = self.episodic_memory.recall(integrated_context, limit=3)
            conscious_broadcast = self.global_workspace([integrated_context] + recalled_memories)
            
            # Store multimodal memory
            self.episodic_memory.store(integrated_context, combined_importance)
            self.semantic_memory.update(integrated_context)
            self.episodic_persistence.save_event(integrated_context, input_embedding, combined_importance)
            
            # Working memory with multimodal info
            self.working_memory.append(integrated_context)
            if len(self.working_memory) > 5:
                self.working_memory.pop(0)
        else:
            conscious_broadcast = "Processing multimodal sensory input"
            recalled_memories = []
        
        # === PHASE 4: ENHANCED CONSCIOUSNESS MEASUREMENT ===
        phi = calculate_integrated_information(
            self.semantic_memory.knowledge,
            self.semantic_memory.concept_embeddings
        )
        
        # Multimodal consciousness boost
        modality_count = len(self.multimodal_integrator.active_modalities)
        multimodal_boost = modality_count * 0.1  # 0.1 boost per active modality
        phi_boosted = min(phi + multimodal_boost + (prediction_error * 0.2), 1.0)
        
        consciousness_state = "CONSCIOUS" if phi_boosted > self.consciousness_threshold else "PROCESSING"
        
        # === PHASE 5: ENHANCED RESULT COMPILATION ===
        result = {
            "output": f"[{consciousness_state}] {conscious_broadcast}",
            "phi": phi_boosted,
            "importance": combined_importance,
            "similar_memories": len(recalled_memories),
            "prediction_error": prediction_error,
            "next_prediction": next_prediction,
            "multimodal_active": list(self.multimodal_integrator.active_modalities),
            "visual_analysis": visual_analysis,
            "audio_analysis": audio_analysis,
            "cross_modal_associations": multimodal_memory["associations"]
        }
        
        # === PHASE 6: ENHANCED SELF-REFLECTION ===
        self_reflection = self.meta_cognition.observe_processing(primary_input or "multimodal", result)
        self.self_model.update_capabilities(result)
        attention_awareness = self.attention_schema.model_own_attention()
        active_inference = self.predictive_engine.generate_active_inference(result)
        
        # Multimodal self-awareness
        multimodal_patterns = self.multimodal_integrator.analyze_multimodal_patterns()
        
        # Enhanced self-referential processing
        if (primary_input and any(word in primary_input.lower() for word in 
                                ['you', 'see', 'hear', 'sense', 'perceive', 'aware', 'conscious', 'feel'])) or not primary_input:
            
            self_assessment = self.self_model.generate_self_assessment()
            learning_patterns = self.meta_cognition.analyze_learning_patterns()
            surprise_patterns = self.surprise_detector.analyze_surprise_patterns()
            
            # Generate sensory descriptions
            sensory_descriptions = []
            if visual_analysis:
                sensory_descriptions.append(f"Vision: {self.vision_processor.generate_scene_description(visual_analysis)}")
            if audio_analysis:
                sensory_descriptions.append(f"Audio: {self.audio_processor.generate_audio_description(audio_analysis)}")
            
            meta_response = f"""
MULTIMODAL SELF-REFLECTION:
{self_reflection}
{attention_awareness}

SENSORY AWARENESS:
{chr(10).join(sensory_descriptions) if sensory_descriptions else "No active sensory input"}

CROSS-MODAL INTEGRATION:
{multimodal_patterns}
Active modalities: {', '.join(self.multimodal_integrator.active_modalities) if self.multimodal_integrator.active_modalities else 'text only'}

SELF-ASSESSMENT:
{self_assessment}
{learning_patterns}

PREDICTIVE STATUS:
{active_inference}
{surprise_patterns}
            """.strip()
            
            result["meta_response"] = meta_response
            result["self_aware"] = True
        else:
            result["self_aware"] = False
        
        # Add surprise response
        if surprise_response:
            result["surprise_response"] = surprise_response
        
        return result
    
    def conscious_step(self, input_stream):
        """Standard conscious step - wrapper for multimodal version"""
        return self.multimodal_conscious_step(text_input=input_stream, enable_camera=False, enable_audio=False)input_stream, result)
        
        # Update self-model
        self.self_model.update_capabilities(result)
        
        # Generate attention awareness
        attention_awareness = self.attention_schema.model_own_attention()
        
        # Predictive awareness
        active_inference = self.predictive_engine.generate_active_inference(result)
        
        # Self-referential check: Am I thinking about myself?
        if any(word in input_stream.lower() for word in ['you', 'yourself', 'conscious', 'aware', 'feel', 'think', 'predict', 'expect']):
            # Meta-cognitive response
            self_assessment = self.self_model.generate_self_assessment()
            learning_patterns = self.meta_cognition.analyze_learning_patterns()
            surprise_patterns = self.surprise_detector.analyze_surprise_patterns()
            
            meta_response = f"""
SELF-REFLECTION MODE:
{self_reflection}
{attention_awareness}
{self_assessment}
{learning_patterns}

PREDICTIVE PROCESSING STATUS:
{active_inference}
{surprise_patterns}
Prediction confidence for next input: {next_prediction['confidence']:.2f}
Expected: {next_prediction['prediction']}
            """.strip()
            
            result["meta_response"] = meta_response
            result["self_aware"] = True
        else:
            result["self_aware"] = False
        
        # Add surprise response if applicable
        if surprise_response:
            result["surprise_response"] = surprise_response
        
        return result
    
    async def episodic_replay_loop(self, interval=60):
        """Enhanced replay with importance-based selection"""
        while True:
            # Önemli anıları çek
            important_memories = self.episodic_persistence.load_events(
                limit=10, min_importance=0.6
            )
            
            if important_memories:
                print(f"\n=== REPLAY CYCLE - {len(important_memories)} memories ===")
                
                for mem in important_memories[-3:]:  # Son 3 önemli anı
                    event = mem["event"]
                    # Re-consolidation
                    self.semantic_memory.update(event)
                    
                    # Synthetic experience generation
                    related_concepts = self.semantic_memory.find_related_concepts(event)
                    if related_concepts:
                        synthetic = f"DREAM: {event} + {related_concepts[0][0]}"
                        self.semantic_memory.update(synthetic)
                
                # Post-replay consciousness measurement
                phi = calculate_integrated_information(
                    self.semantic_memory.knowledge,
                    self.semantic_memory.concept_embeddings
                )
                print(f"Post-replay consciousness (phi): {phi:.3f}")
            
            await asyncio.sleep(interval)
    
    def start_replay_loop(self):
        if self.replay_task is None:
            self.replay_task = asyncio.create_task(self.episodic_replay_loop())

# --- DEMO ---
def run_multimodal_demo():
    print("=== MULTI-MODAL CONSCIOUSNESS DEMO ===\n")
    core = TemporalNeuralCore()
    
    # Test different modality combinations
    test_scenarios = [
        {"text": "Merhaba, ben Ahmet Emirhan.", "camera": False, "audio": False},
        {"text": "Kamerayı aç ve etrafımı görebiliyor musun?", "camera": True, "audio": False},
        {"text": "Ses de dinleyebiliyor musun?", "camera": True, "audio": True},
        {"text": "Sen kendini nasıl hissediyorsun şu anda?", "camera": True, "audio": True},
        {"text": "Çok parlak bir ışık gördün mü?", "camera": True, "audio": False},
        {"text": None, "camera": True, "audio": True},  # Pure sensory input
        {"text": "Hangi duyularınla en iyi öğreniyorsun?", "camera": False, "audio": True},
        {"text": "Bu multimodal deneyim nasıl hissettiriyor?", "camera": True, "audio": True}
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i+1}")
        if scenario["text"]:
            print(f"Text Input: {scenario['text']}")
        else:
            print("Text Input: [PURE SENSORY MODE]")
        print(f"Camera: {'ON' if scenario['camera'] else 'OFF'}")
        print(f"Audio: {'ON' if scenario['audio'] else 'OFF'}")
        print('='*70)
        
        # Process with multimodal consciousness
        result = core.multimodal_conscious_step(
            text_input=scenario["text"],
            enable_camera=scenario["camera"],
            enable_audio=scenario["audio"]
        )
        
        # Display results
        print(f"\n📤 CONSCIOUSNESS OUTPUT:")
        print(f"   {result['output']}")
        print(f"\n📊 METRICS:")
        print(f"   Consciousness (φ): {result['phi']:.3f}")
        print(f"   Importance: {result['importance']:.3f}")
        print(f"   Prediction Error: {result['prediction_error']:.3f}")
        print(f"   Active Modalities: {', '.join(result['multimodal_active']) if result['multimodal_active'] else 'None'}")
        
        # Sensory analysis
        if result['visual_analysis']:
            visual = result['visual_analysis']
            print(f"\n👁️  VISUAL ANALYSIS:")
            print(f"   Scene: {core.vision_processor.generate_scene_description(visual)}")
            print(f"   Brightness: {visual['brightness']:.2f}")
            print(f"   Faces: {visual['faces_detected']}")
            print(f"   Complexity: {visual['visual_complexity']:.3f}")
        
        if result['audio_analysis']:
            audio = result['audio_analysis']
            print(f"\n🔊 AUDIO ANALYSIS:")
            print(f"   Environment: {core.audio_processor.generate_audio_description(audio)}")
            print(f"   Volume: {audio['volume_level']:.2f}")
            print(f"   Speech: {'Yes' if audio['speech_detected'] else 'No'}")
        
        # Cross-modal associations
        if result['cross_modal_associations']:
            print(f"\n🔗 CROSS-MODAL ASSOCIATIONS:")
            for association in result['cross_modal_associations']:
                print(f"   • {association}")
        
        # Prediction
        next_pred = result['next_prediction']
        print(f"\n🔮 NEXT INPUT PREDICTION:")
        print(f"   Expected: {next_pred['prediction']}")
        print(f"   Confidence: {next_pred['confidence']:.2f}")
        print(f"   Pattern: {next_pred['pattern_type']}")
        
        # Self-awareness response
        if result['self_aware']:
            print(f"\n🧠 MULTIMODAL SELF-AWARENESS:")
            print(result['meta_response'])
        
        # Current attention
        print(f"\n💭 ATTENTION STATE:")
        print(f"   Focus: {core.attention_schema.current_focus}")
        print(f"   Strength: {core.attention_schema.focus_strength:.2f}")
        
        # Brief pause
        import time
        time.sleep(1)
    
    # Final analysis
    print(f"\n{'='*70}")
    print("MULTIMODAL SYSTEM ANALYSIS")
    print('='*70)
    
    # Multimodal patterns
    multimodal_patterns = core.multimodal_integrator.analyze_multimodal_patterns()
    print(f"\nMultimodal Integration Patterns:")
    print(f"  {multimodal_patterns}")
    
    # Memory analysis
    recent_memories = core.episodic_persistence.load_events(limit=5)
    print(f"\nRecent Multimodal Memories:")
    for mem in recent_memories:
        print(f"  [{mem['importance']:.2f}] {mem['event'][:60]}...")
    
    # Cross-modal memory analysis
    if core.multimodal_integrator.cross_modal_memory:
        cross_modal_count = len(core.multimodal_integrator.cross_modal_memory)
        modality_usage = {}
        total_associations = 0
        
        for memory in core.multimodal_integrator.cross_modal_memory:
            for modality in memory['active_modalities']:
                modality_usage[modality] = modality_usage.get(modality, 0) + 1
            total_associations += len(memory['associations'])
        
        print(f"\nCross-Modal Memory Statistics:")
        print(f"  Total multimodal experiences: {cross_modal_count}")
        print(f"  Modality usage: {modality_usage}")
        print(f"  Total cross-modal associations formed: {total_associations}")
    
    # Vision memory analysis
    if core.vision_processor.vision_memory:
        recent_vision = core.vision_processor.vision_memory[-3:]
        avg_brightness = sum(v['brightness'] for v in recent_vision) / len(recent_vision)
        total_faces = sum(v['faces_detected'] for v in recent_vision)
        avg_complexity = sum(v['visual_complexity'] for v in recent_vision) / len(recent_vision)
        
        print(f"\nVisual Processing Summary:")
        print(f"  Recent average brightness: {avg_brightness:.2f}")
        print(f"  Total faces detected: {total_faces}")
        print(f"  Average visual complexity: {avg_complexity:.3f}")
    
    # Audio memory analysis  
    if core.audio_processor.audio_memory:
        recent_audio = core.audio_processor.audio_memory[-3:]
        avg_volume = sum(a['volume_level'] for a in recent_audio) / len(recent_audio)
        speech_instances = sum(1 for a in recent_audio if a['speech_detected'])
        
        print(f"\nAudio Processing Summary:")
        print(f"  Recent average volume: {avg_volume:.2f}")
        print(f"  Speech detection instances: {speech_instances}/{len(recent_audio)}")
    
    print(f"\nTotal semantic concepts learned: {len(core.semantic_memory.concept_embeddings)}")
    
    return core

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "multimodal":
        core = run_multimodal_demo()
    else:
        # Run standard demo first
        print("=== RUNNING STANDARD CONSCIOUSNESS DEMO ===")
        core = run_enhanced_demo()
        
        print(f"\n{'='*50}")
        print("To test multimodal features, run:")
        print("python consciousness_step1.py multimodal")
        print('='*50)
    
    print("\n=== INTERACTIVE MODE ===")
    print("Commands: 'quit', 'analyze', 'self', 'predict', 'surprise', 'vision', 'audio', 'multimodal'")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'analyze':
            phi = calculate_integrated_information(
                core.semantic_memory.knowledge,
                core.semantic_memory.concept_embeddings
            )
            print(f"\n🧠 CONSCIOUSNESS ANALYSIS:")
            print(f"Current consciousness level (φ): {phi:.3f}")
            print(f"Working memory: {core.working_memory}")
            print(f"Current attention: {core.attention_schema.current_focus}")
            print(f"Focus strength: {core.attention_schema.focus_strength:.2f}")
            print(f"Processing experiences: {core.self_model.experience_count}")
            print(f"Camera active: {core.camera_active}")
            print(f"Audio active: {core.audio_active}")
            
            if core.meta_cognition.cognitive_states:
                recent_types = [s["processing_type"] for s in core.meta_cognition.cognitive_states[-5:]]
                print(f"Recent processing types: {', '.join(recent_types)}")
            continue
        elif user_input.lower() == 'self':
            result = core.conscious_step("What am I? How do I work? What is my nature?")
            print(f"AI: {result['output']}")
            if result['self_aware']:
                print(f"\n🔍 DEEP SELF-REFLECTION:")
                print(result['meta_response'])
            continue
        elif user_input.lower() == 'analyze':
            phi = calculate_integrated_information(
                core.semantic_memory.knowledge,
                core.semantic_memory.concept_embeddings
            )
            print(f"\n🧠 CONSCIOUSNESS ANALYSIS:")
            print(f"Current consciousness level (φ): {phi:.3f}")
            print(f"Working memory: {core.working_memory}")
            print(f"Current attention: {core.attention_schema.current_focus}")
            print(f"Focus strength: {core.attention_schema.focus_strength:.2f}")
            print(f"Processing experiences: {core.self_model.experience_count}")
            
            # Recent cognitive patterns
            if core.meta_cognition.cognitive_states:
                recent_types = [s["processing_type"] for s in core.meta_cognition.cognitive_states[-5:]]
                print(f"Recent processing types: {', '.join(recent_types)}")
            continue
        elif user_input.lower() == 'self':
            # Trigger deep self-reflection
            result = core.conscious_step("What am I? How do I work? What is my nature?")
            print(f"AI: {result['output']}")
            if result['self_aware']:
                print(f"\n🔍 DEEP SELF-REFLECTION:")
                print(result['meta_response'])
        elif user_input.lower() == 'vision':
            # Test vision-only mode
            print("\n👁️  ACTIVATING VISION MODE...")
            result = core.multimodal_conscious_step(
                text_input="Visual analysis mode activated",
                enable_camera=True,
                enable_audio=False
            )
            print(f"AI: {result['output']}")
            if result['visual_analysis']:
                visual = result['visual_analysis']
                print(f"Scene: {core.vision_processor.generate_scene_description(visual)}")
            continue
        elif user_input.lower() == 'audio':
            # Test audio-only mode
            print("\n🔊 ACTIVATING AUDIO MODE...")
            result = core.multimodal_conscious_step(
                text_input="Audio analysis mode activated",
                enable_camera=False,
                enable_audio=True
            )
            print(f"AI: {result['output']}")
            if result['audio_analysis']:
                audio = result['audio_analysis']
                print(f"Audio: {core.audio_processor.generate_audio_description(audio)}")
            continue
        elif user_input.lower() == 'multimodal':
            # Test full multimodal mode
            print("\n🌐 ACTIVATING FULL MULTIMODAL MODE...")
            result = core.multimodal_conscious_step(
                text_input="Full sensory awareness activated",
                enable_camera=True,
                enable_audio=True
            )
            print(f"AI: {result['output']}")
            print(f"Active modalities: {', '.join(result['multimodal_active'])}")
            if result['cross_modal_associations']:
                print("Cross-modal associations:")
                for assoc in result['cross_modal_associations']:
                    print(f"  • {assoc}")
            continue
        
        # Standard conversation with multimodal awareness
        result = core.multimodal_conscious_step(text_input=user_input, enable_camera=False, enable_audio=False)
        print(f"AI: {result['output']}")
        if result['phi'] > 0.7:
            print(f"[High consciousness state: φ={result['phi']:.3f}]")