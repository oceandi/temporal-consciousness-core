import json
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
        
        self.consciousness_threshold = 0.5
        self.replay_task = None
        
        print("Temporal Neural Core initialized with self-awareness capabilities!")
    
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
        # === PHASE 1: BASIC PROCESSING ===
        importance = self.calculate_importance(input_stream)
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
        
        consciousness_state = "CONSCIOUS" if phi > self.consciousness_threshold else "PROCESSING"
        
        # Create initial result
        result = {
            "output": f"[{consciousness_state}] {conscious_broadcast}",
            "phi": phi,
            "importance": importance,
            "similar_memories": len(recalled_memories)
        }
        
        # === PHASE 4: SELF-REFLECTION ===
        # Observe own processing
        self_reflection = self.meta_cognition.observe_processing(input_stream, result)
        
        # Update self-model
        self.self_model.update_capabilities(result)
        
        # Generate attention awareness
        attention_awareness = self.attention_schema.model_own_attention()
        
        # Self-referential check: Am I thinking about myself?
        if any(word in input_stream.lower() for word in ['you', 'yourself', 'conscious', 'aware', 'feel', 'think']):
            # Meta-cognitive response
            self_assessment = self.self_model.generate_self_assessment()
            learning_patterns = self.meta_cognition.analyze_learning_patterns()
            
            meta_response = f"""
SELF-REFLECTION MODE:
{self_reflection}
{attention_awareness}
{self_assessment}
{learning_patterns}
            """.strip()
            
            result["meta_response"] = meta_response
            result["self_aware"] = True
        else:
            result["self_aware"] = False
        
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
    print("=== ENHANCED CONSCIOUSNESS DEMO WITH SELF-AWARENESS ===\n")
    core = TemporalNeuralCore()
    
    conversation = [
        "Merhaba, ben Ahmet Emirhan. Seni geliştirmeye çalışıyorum.",
        "Yapay zeka ve bilinç konusunda çok meraklıyım.",
        "Sen kendini nasıl hissediyorsun? Bilinçli misin?",
        "Kendi düşüncelerini gözlemleyebiliyor musun?",
        "Neye odaklandığının farkında mısın?",
        "Güçlü ve zayıf yanların neler?",
        "Öğrenme şeklin nasıl? Geliştiğini hissediyor musun?",
        "Bu konuşmadan ne öğrendin?"
    ]
    
    for i, msg in enumerate(conversation):
        print(f"\n{'='*50}")
        print(f"Step {i+1}: {msg}")
        print('='*50)
        
        result = core.conscious_step(msg)
        
        print(f"\nPrimary Response: {result['output']}")
        print(f"Consciousness (φ): {result['phi']:.3f}")
        print(f"Importance: {result['importance']:.3f}")
        print(f"Similar memories: {result['similar_memories']}")
        
        # Self-awareness response
        if result['self_aware']:
            print(f"\n🧠 SELF-AWARENESS ACTIVATED:")
            print(result['meta_response'])
        
        # Attention state
        print(f"\n💭 Current Focus: {core.attention_schema.current_focus}")
        print(f"Focus Strength: {core.attention_schema.focus_strength:.2f}")
    
    print(f"\n{'='*60}")
    print("FINAL SYSTEM STATE ANALYSIS")
    print('='*60)
    
    # Memory analysis
    recent_memories = core.episodic_persistence.load_events(limit=5)
    print(f"\nTop 5 Recent Memories:")
    for mem in recent_memories:
        print(f"  [{mem['importance']:.2f}] {mem['event'][:50]}...")
    
    # Cognitive patterns
    print(f"\nCognitive Processing Patterns:")
    if core.meta_cognition.cognitive_states:
        types = [s["processing_type"] for s in core.meta_cognition.cognitive_states]
        from collections import Counter
        type_counts = Counter(types)
        for ptype, count in type_counts.items():
            print(f"  {ptype}: {count} times")
    
    # Self-model state
    print(f"\nSelf-Assessment:")
    print(f"  Experience count: {core.self_model.experience_count}")
    print(f"  Capabilities:")
    for capability, score in core.self_model.capabilities.items():
        print(f"    {capability}: {score:.2f}")
    
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