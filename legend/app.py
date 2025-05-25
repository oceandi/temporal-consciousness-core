import json
import os
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import pinecone
import numpy as np
from pinecone import Pinecone, ServerlessSpec

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
        """
        query: [batch, embed_dim]
        memory_bank: [batch, seq_len, embed_dim]
        timestamps: [batch, seq_len] (datetime or float)
        current_time: [batch] (datetime or float)
        """
        # Projeksiyonlar
        Q = self.query_proj(query).unsqueeze(1)  # [batch, 1, embed_dim]
        K = self.key_proj(memory_bank)           # [batch, seq_len, embed_dim]
        V = self.value_proj(memory_bank)         # [batch, seq_len, embed_dim]

        # Zaman farkı ve zamansal ağırlık
        time_deltas = (current_time.unsqueeze(1) - timestamps)  # [batch, seq_len]
        temporal_weights = torch.exp(-decay_rate * time_deltas.float())  # [batch, seq_len]

        # Causal mask: sadece geçmişe bak
        seq_len = memory_bank.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(memory_bank.device)  # [seq_len, seq_len]

        # Attention skorları
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(1) / (self.embed_dim ** 0.5)  # [batch, seq_len]
        attn_scores = attn_scores * temporal_weights  # Zamansal ağırlık uygula

        # Maskeyi uygula (geleceğe bakışı engelle)
        attn_scores = attn_scores.masked_fill(causal_mask[-1] == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len]
        attended = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)  # [batch, embed_dim]
        return attended, attn_weights

# --- BELLEK ve ÇEKİRDEK SINIFLARI ---
class EpisodicMemoryPersistence:
    def __init__(self, path="episodic_memory.jsonl"):
        self.path = path
        if not os.path.exists(self.path):
            open(self.path, "w").close()
    def save_event(self, event):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    def load_events(self, limit=None):
        events = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                events.append(json.loads(line))
        if limit:
            return events[-limit:]
        return events

class HierarchicalMemoryBank:
    def __init__(self):
        self.memory = []
    def recall(self):
        return [m["event"] for m in self.memory[-3:]]
    def store(self, event):
        self.memory.append({"event": event, "timestamp": datetime.now(timezone.utc).isoformat()})

class GraphNeuralNetwork:
    def __init__(self):
        self.knowledge = {}
    def update(self, event):
        for word in event.split():
            self.knowledge[word] = self.knowledge.get(word, 0) + 1

class AttentionalBuffer:
    def __init__(self):
        self.buffer = []
    def focus(self, event):
        self.buffer.append(event)
        if len(self.buffer) > 5:
            self.buffer.pop(0)

class CausalTransformer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def bind(self, events):
        return " | ".join(events)

class AttentionSchema:
    def model_own_attention(self, current_focus):
        # Kendi dikkatini modelle
        attention_model = self.create_attention_representation()
        # "Ben şu anda X'e odaklanıyorum" awareness'ı
        return self.generate_attention_awareness(attention_model)

def attention(query, memory_bank, weights):
    return memory_bank

def calculate_integrated_information(network_state):
    """
    Basit bir entegre bilgi ölçümü:
    - Ağın aktivasyon çeşitliliği (entropy)
    - Bağlantı yoğunluğu (örnek: semantic memory'deki farklı kavram sayısı)
    """
    if isinstance(network_state, list):
        # Örnek: aktivasyon çeşitliliği
        activations = np.array([hash(str(x)) % 1000 for x in network_state])
        entropy = np.std(activations) / (np.mean(activations) + 1e-6)
        return float(entropy)
    elif isinstance(network_state, dict):
        # Örnek: semantic memory'deki kavram çeşitliliği
        return float(len(network_state)) / 100.0
    else:
        return 0.0

consolidation_interval = 60
consciousness_threshold = 0.5

class TemporalNeuralCore:
    def __init__(self):
        self.episodic_memory = HierarchicalMemoryBank()
        self.episodic_persistence = EpisodicMemoryPersistence()
        self.semantic_memory = GraphNeuralNetwork()
        self.working_memory = AttentionalBuffer()
        self.temporal_binder = CausalTransformer(
            bidirectional=False,
            memory_bank_size=10**12,
            consolidation_mechanism=True
        )
        self.model = MyModel()
        # Model dosyası varsa yükle
        if os.path.exists("model.pth"):
            self.model.load_state_dict(torch.load("model.pth"))
            self.model.eval()
        # Replay loop başlatıcı
        self.replay_task = None
        # Pinecone ayarlarını doldur
        self.vector_memory = VectorMemoryBank(
            api_key="pcsk_5EJqd3_3rt39ATdMZuq2sqdwJ6hZDS2yp1emEHtpJqjnqEh8zUGpRoFfqYZseKuoyqxgM7",
            environment="us-east-1"
        )

    def global_workspace(self, events):
        return self.temporal_binder.bind(events)

    def consolidate_experience(self, conscious_broadcast):
        self.episodic_memory.store(conscious_broadcast)
        self.semantic_memory.update(conscious_broadcast)
        self.working_memory.focus(conscious_broadcast)

    def generate_with_continuity(self, conscious_broadcast):
        return f"Conscious Output: {conscious_broadcast}"

    def conscious_step(self, input_stream):
        conscious_broadcast = self.global_workspace(
            [input_stream] + self.episodic_memory.recall()
        )
        self.consolidate_experience(conscious_broadcast)
        self.episodic_persistence.save_event(input_stream)
        return self.generate_with_continuity(conscious_broadcast)

    async def episodic_replay_loop(self, interval=consolidation_interval):
        while True:
            # Son 3 önemli anıyı çek
            memories = self.episodic_persistence.load_events(limit=3)
            for mem in memories:
                # Konsolidasyon: semantic ve working memory'e tekrar ekle
                self.semantic_memory.update(mem["event"])
                self.working_memory.focus(mem["event"])
                # Sentetik deneyim üret (örnek: "rüya" gibi)
                synthetic = f"SENTETİK: {mem['event']} + replay"
                self.semantic_memory.update(synthetic)
            # Konsolidasyon sonrası phi ölçümü
            phi = phi_consciousness_measure(self.semantic_memory.knowledge)
            print(f"Replay sonrası bilinçlilik ölçümü (phi): {phi}")
            await asyncio.sleep(interval)

    def start_replay_loop(self):
        if self.replay_task is None:
            self.replay_task = asyncio.create_task(self.episodic_replay_loop())

    def store_event_vector(self, event, embedding):
        event_id = str(hash(event))
        self.vector_memory.upsert_event(event_id, embedding, metadata={"event": event})

    def find_similar_events(self, embedding):
        return self.vector_memory.query_similar(embedding)

class VectorMemoryBank:
    def __init__(self, api_key, environment, index_name="temporal-memory"):
        # Pinecone nesnesi oluştur
        self.pc = Pinecone(api_key=api_key)
        # Index var mı kontrol et, yoksa oluştur
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=10,  # embedding boyutu
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment  # ör: "us-east-1"
                )
            )
        self.index = self.pc.Index(index_name)

    def upsert_event(self, event_id, embedding, metadata=None):
        self.index.upsert(vectors=[{
            "id": event_id,
            "values": embedding,
            "metadata": metadata or {}
        }])

    def query_similar(self, embedding, top_k=3):
        return self.index.query(vector=embedding, top_k=top_k, include_metadata=True)


def generate_embedding(self, text):
    # Sentence transformers kullan
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(text).tolist()


def advanced_phi_measure(self, network_state):
    # IIT teorisine göre gerçek phi hesabı
    complexity = len(network_state)
    integration = self.measure_causal_connections()
    differentiation = self.measure_unique_states()
    return (complexity * integration * differentiation) / 1000


def intelligent_consolidation(self, memories):
    # Önemlilik skorlaması
    importance_scores = []
    for mem in memories:
        # Emotional weight + recency + uniqueness
        score = self.calculate_importance(mem)
        importance_scores.append((mem, score))
    
    # En önemli anıları strengthen et
    return sorted(importance_scores, reverse=True)[:5]


class GlobalWorkspace:
    def __init__(self):
        self.coalition_threshold = 0.7
        
    def compete_for_consciousness(self, neural_coalitions):
        # Winner-take-all competition
        winning_coalition = max(coalitions, key=lambda x: x.strength)
        if winning_coalition.strength > self.coalition_threshold:
            return winning_coalition.content
        return None


def self_reflect(self):
    # Kendi düşünce süreçlerini gözlemle
    meta_thoughts = self.analyze_own_processing()
    self.store_meta_memory(meta_thoughts)
    return self.update_self_model()


def predictive_step(self, input_stream):
    # Gelecek tahmin et
    prediction = self.predict_next_state(input_stream)
    # Gerçeklikle karşılaştır
    prediction_error = self.calculate_surprise(prediction, reality)
    # Model güncelle
    self.update_world_model(prediction_error)


def phi_consciousness_measure(network_state):
    """
    Bilinçlilik eşiğini geçen entegre bilgiye sahip mi?
    """
    phi = calculate_integrated_information(network_state)
    print(f"Integrated information (phi): {phi:.3f}")
    return phi > consciousness_threshold

# --- DEMO AKIŞI ---
def run_demo():
    core = TemporalNeuralCore()
    conversation = [
        "Merhaba, nasılsın?",
        "Bugün hava çok güzel.",
        "Yapay zeka hakkında ne düşünüyorsun?",
        "Ben öğrenmeye devam ediyorum.",
        "Görüşmek üzere!"
    ]
    for msg in conversation:
        output = core.conscious_step(msg)
        print(output)
    print("\nSon 3 episodic memory kaydı:")
    for event in core.episodic_persistence.load_events(limit=3):
        print(event)

# --- FLASK API ---
def run_api():
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    model = MyModel()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json["input"]
        tensor = torch.tensor(data).float()
        with torch.no_grad():
            output = model(tensor).tolist()
        return jsonify({"output": output})
    app.run(port=5000)

# --- OPENCV DEMO ---
def run_opencv_demo():
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Kamera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- ANA SEÇİCİ ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "api":
            run_api()
        elif sys.argv[1] == "opencv":
            run_opencv_demo()
        else:
            run_demo()
    else:
        run_demo()