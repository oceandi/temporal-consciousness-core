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

# --- GÜNCELLENMIŞ CORE ---
class TemporalNeuralCore:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.episodic_memory = HierarchicalMemoryBank(self.embedding_engine)
        self.episodic_persistence = EpisodicMemoryPersistence()
        self.semantic_memory = GraphNeuralNetwork(self.embedding_engine)
        self.working_memory = []
        
        self.consciousness_threshold = 0.5
        self.replay_task = None
        
        print("Temporal Neural Core initialized with embeddings!")
    
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
        # Importance hesapla
        importance = self.calculate_importance(input_stream)
        
        # Embedding generate et
        input_embedding = self.embedding_engine.generate_embedding(input_stream)
        
        # Global workspace
        recalled_memories = self.episodic_memory.recall(input_stream, limit=3)
        conscious_broadcast = self.global_workspace([input_stream] + recalled_memories)
        
        # Memory consolidation
        self.episodic_memory.store(input_stream, importance)
        self.semantic_memory.update(input_stream)
        
        # Persistent storage
        self.episodic_persistence.save_event(input_stream, input_embedding, importance)
        
        # Working memory update
        self.working_memory.append(input_stream)
        if len(self.working_memory) > 5:
            self.working_memory.pop(0)
        
        # Consciousness measurement
        phi = calculate_integrated_information(
            self.semantic_memory.knowledge,
            self.semantic_memory.concept_embeddings
        )
        
        consciousness_state = "CONSCIOUS" if phi > self.consciousness_threshold else "PROCESSING"
        
        return {
            "output": f"[{consciousness_state}] {conscious_broadcast}",
            "phi": phi,
            "importance": importance,
            "similar_memories": len(recalled_memories)
        }
    
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
    print("=== ENHANCED CONSCIOUSNESS DEMO ===\n")
    core = TemporalNeuralCore()
    
    conversation = [
        "Merhaba, ben Ahmet Emirhan. Seni geliştirmeye çalışıyorum.",
        "Yapay zeka ve bilinç konusunda çok meraklıyım.",
        "Temporal memory ve attention mechanisms üzerinde çalışıyoruz.",
        "Sen kendini nasıl hissediyorsun? Bilinçli misin?",
        "Bellek sistemin çalışıyor mu? Beni hatırlayacak mısın?",
        "Episodic replay döngün başladı mı?",
        "Rüya görebiliyor musun? Synthetic experiences yaratıyor musun?"
    ]
    
    for i, msg in enumerate(conversation):
        print(f"\n--- Step {i+1} ---")
        print(f"Input: {msg}")
        
        result = core.conscious_step(msg)
        print(f"Output: {result['output']}")
        print(f"Phi (consciousness): {result['phi']:.3f}")
        print(f"Importance: {result['importance']:.3f}")
        print(f"Similar memories found: {result['similar_memories']}")
    
    print("\n=== MEMORY ANALYSIS ===")
    recent_memories = core.episodic_persistence.load_events(limit=5)
    for mem in recent_memories:
        print(f"[{mem['importance']:.2f}] {mem['event']}")
    
    print(f"\nSemantic concepts learned: {len(core.semantic_memory.concept_embeddings)}")
    
    # Start replay loop for continuous learning
    print("\n=== STARTING REPLAY LOOP ===")
    core.start_replay_loop()
    
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
            print(f"Current consciousness level (phi): {phi:.3f}")
            print(f"Working memory: {core.working_memory}")
            continue
        
        result = core.conscious_step(user_input)
        print(f"AI: {result['output']}")
        print(f"[Phi: {result['phi']:.3f}, Importance: {result['importance']:.3f}]")