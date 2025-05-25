print("\n=== INTERACTIVE MODE ===")
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
        
        self.consciousness_threshold = 0.5
        self.replay_task = None
        
        print("Temporal Neural Core initialized with predictive processing capabilities!")
    
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
    
    def conscious_step(self, input_stream):
        # === PHASE 0: PREDICTIVE PROCESSING ===
        prediction_error = 0.5  # Default
        surprise_response = None
        
        # Check prediction accuracy if we had a previous prediction
        if self.last_prediction and len(self.conversation_history) > 0:
            prediction_error = self.predictive_engine.calculate_prediction_error(
                self.last_prediction, input_stream
            )
            
            # Process surprise
            surprise_response = self.surprise_detector.process_surprise(prediction_error, input_stream)
            
            # Update world model based on prediction error
            context = self.conversation_history[-1] if self.conversation_history else ""
            was_surprising = self.predictive_engine.update_world_model(context, input_stream, prediction_error)
            
            if was_surprising:
                print(f"🚨 SURPRISE DETECTED: Error={prediction_error:.2f}")
        
        # Generate prediction for NEXT input
        current_context = input_stream
        next_prediction = self.predictive_engine.predict_next_input(current_context, self.conversation_history)
        self.last_prediction = next_prediction
        
        # Add to conversation history
        self.conversation_history.append(input_stream)
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)
        
        # === PHASE 1: BASIC PROCESSING ===
        importance = self.calculate_importance(input_stream)
        
        # Adjust importance based on prediction error (surprising = more important)
        importance = min(importance + (prediction_error * 0.3), 1.0)
        
        input_embedding = self.embedding_engine.generate_embedding(input_stream)
        
        # Update attention schema
        attention_strength = min(importance * 2, 1.0)
        self.attention_schema.update_focus(input_stream, attention_strength)
        
        # === PHASE 2: MEMORY & INTEGRATION ===
        recalled_memories = self.episodic_memory.recall(input_stream, limit=3)
        conscious_broadcast = self.global_workspace([input_stream] + recalled_memories)
        
        # Memory consolidation
        self.episodic_memory.store(input_stream, importance)
        self.semantic_memory.update(input_stream)
        self.episodic_persistence.save_event(input_stream, input_embedding, importance)
        
        # Working memory update
        self.working_memory.append(input_stream)
        if len(self.working_memory) > 5:
            self.working_memory.pop(0)
        
        # === PHASE 3: CONSCIOUSNESS MEASUREMENT ===
        phi = calculate_integrated_information(
            self.semantic_memory.knowledge,
            self.semantic_memory.concept_embeddings
        )
        
        # Consciousness boost from prediction errors (surprise enhances awareness)
        phi_boosted = min(phi + (prediction_error * 0.2), 1.0)
        
        consciousness_state = "CONSCIOUS" if phi_boosted > self.consciousness_threshold else "PROCESSING"
        
        # Create initial result
        result = {
            "output": f"[{consciousness_state}] {conscious_broadcast}",
            "phi": phi_boosted,
            "importance": importance,
            "similar_memories": len(recalled_memories),
            "prediction_error": prediction_error,
            "next_prediction": next_prediction
        }
        
        # === PHASE 4: SELF-REFLECTION ===
        # Observe own processing
        self_reflection = self.meta_cognition.observe_processing(input_stream, result)
        
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
def run_enhanced_demo():
    print("=== CONSCIOUSNESS DEMO WITH PREDICTIVE PROCESSING ===\n")
    core = TemporalNeuralCore()
    
    conversation = [
        "Merhaba, ben Ahmet Emirhan. Seni geliştirmeye çalışıyorum.",
        "Yapay zeka ve bilinç konusunda çok meraklıyım.",
        "Sen kendini nasıl hissediyorsun? Bilinçli misin?",
        "Bir sonraki sorumu tahmin edebilir misin?",
        "Kendi düşüncelerini gözlemleyebiliyor musun?",
        "Bu beklenmedik bir soru: Favori rengin ne?",
        "Tekrar normal konuya dönersek, öğrenme şeklin nasıl?",
        "Tahminlerinin ne kadar doğru olduğunu merak ediyorum.",
        "Son olarak, geliştiğini hissediyor musun?"
    ]
    
    for i, msg in enumerate(conversation):
        print(f"\n{'='*60}")
        print(f"Step {i+1}: {msg}")
        print('='*60)
        
        result = core.conscious_step(msg)
        
        print(f"\n📤 Primary Response: {result['output']}")
        print(f"🧠 Consciousness (φ): {result['phi']:.3f}")
        print(f"⭐ Importance: {result['importance']:.3f}")
        print(f"💾 Similar memories: {result['similar_memories']}")
        print(f"⚠️  Prediction error: {result['prediction_error']:.3f}")
        
        # Show prediction for next input
        next_pred = result['next_prediction']
        print(f"\n🔮 PREDICTION FOR NEXT INPUT:")
        print(f"   Expected: {next_pred['prediction']}")
        print(f"   Confidence: {next_pred['confidence']:.2f}")
        print(f"   Pattern: {next_pred['pattern_type']}")
        print(f"   Reasoning: {next_pred['reasoning']}")
        
        # Surprise response
        if 'surprise_response' in result:
            surprise = result['surprise_response']
            print(f"\n😲 SURPRISE DETECTED:")
            print(f"   Level: {surprise['surprise_level']}")
            print(f"   Response: {surprise['surprise_response']}")
            print(f"   Error magnitude: {surprise['error_magnitude']:.3f}")
        
        # Self-awareness response
        if result['self_aware']:
            print(f"\n🧠 SELF-AWARENESS ACTIVATED:")
            print(result['meta_response'])
        
        # Current attention state
        print(f"\n💭 Attention: {core.attention_schema.current_focus}")
        print(f"    Strength: {core.attention_schema.focus_strength:.2f}")
        
        # Brief pause for readability
        import time
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print("FINAL SYSTEM STATE ANALYSIS")
    print('='*60)
    
    # Memory analysis
    recent_memories = core.episodic_persistence.load_events(limit=5)
    print(f"\nTop 5 Recent Memories:")
    for mem in recent_memories:
        print(f"  [{mem['importance']:.2f}] {mem['event'][:50]}...")
    
    # Prediction accuracy analysis
    if hasattr(core.predictive_engine, 'world_model') and core.predictive_engine.world_model.get('contexts'):
        contexts = core.predictive_engine.world_model['contexts']
        errors = [ctx['error'] for ctx in contexts[-5:]]
        avg_error = sum(errors) / len(errors) if errors else 0.5
        accuracy = (1 - avg_error) * 100
        print(f"\nPrediction Performance:")
        print(f"  Recent accuracy: {accuracy:.1f}%")
        print(f"  Total predictions made: {len(contexts)}")
        
        # Pattern statistics
        pattern_stats = core.predictive_engine.world_model.get('pattern_stats', {})
        if pattern_stats:
            print(f"  Pattern Recognition:")
            for pattern, stats in pattern_stats.items():
                pattern_accuracy = (1 - stats['avg_error']) * 100
                print(f"    {pattern}: {pattern_accuracy:.1f}% accurate ({stats['count']} samples)")
    
    # Surprise analysis
    if core.surprise_detector.surprise_history:
        surprises = core.surprise_detector.surprise_history
        high_surprises = [s for s in surprises if s['level'] == 'high_surprise']
        print(f"\nSurprise Analysis:")
        print(f"  Total surprises: {len(surprises)}")
        print(f"  High surprises: {len(high_surprises)}")
        if high_surprises:
            print(f"  Most surprising: '{high_surprises[-1]['input'][:40]}...' (error: {high_surprises[-1]['error']:.2f})")
    
    print(f"\nSemantic concepts learned: {len(core.semantic_memory.concept_embeddings)}")
    
    return core

if __name__ == "__main__":
    core = run_enhanced_demo()
    
    # Interactive mode
    print("\n=== INTERACTIVE MODE ===")
    print("Type 'quit' to exit, 'analyze' for memory analysis")
    
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
        
        result = core.conscious_step(user_input)
        print(f"AI: {result['output']}")
        print(f"[Phi: {result['phi']:.3f}, Importance: {result['importance']:.3f}]")