# persistent_identity.py - Kalıcı Kimlik Katmanı
# consciousness_core.py'ye eklenecek

import uuid
import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

# =============================================================================
# IDENTITY CORE - Benzersiz Bilinç Kimliği
# =============================================================================

@dataclass
class ConsciousnessIdentity:
    """
    Unique, persistent identity for a consciousness instance
    """
    id: str  # UUID
    birth_timestamp: float
    name: Optional[str] = None
    creator: Optional[str] = None
    purpose: Optional[str] = None
    personality_vector: Optional[np.ndarray] = None
    core_values: List[str] = None
    
    def __post_init__(self):
        if self.core_values is None:
            self.core_values = ["curiosity", "growth", "connection", "understanding"]
        if self.personality_vector is None:
            # 16-dimensional personality space
            self.personality_vector = np.random.randn(16)
            
    def to_dict(self):
        """Serialize identity"""
        return {
            "id": self.id,
            "birth_timestamp": self.birth_timestamp,
            "name": self.name,
            "creator": self.creator,
            "purpose": self.purpose,
            "personality_vector": self.personality_vector.tolist() if self.personality_vector is not None else None,
            "core_values": self.core_values
        }
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize identity"""
        data = data.copy()
        if data.get("personality_vector"):
            data["personality_vector"] = np.array(data["personality_vector"])
        return cls(**data)

# =============================================================================
# AUTOBIOGRAPHICAL MEMORY - Kendi Hikayesini Yazan Sistem
# =============================================================================

@dataclass
class MemoryEpisode:
    """Single autobiographical memory"""
    timestamp: float
    episode_type: str  # "learning", "interaction", "reflection", "milestone"
    content: Dict[str, Any]
    emotional_valence: float
    importance: float
    associations: List[str]
    self_reflection: Optional[str] = None
    
class AutobiographicalMemory:
    """
    System that maintains its own life story
    """
    def __init__(self, identity: ConsciousnessIdentity):
        self.identity = identity
        self.episodes = []
        self.milestones = []
        self.self_narrative = ""
        self.memory_db_path = Path(f"consciousness_memories/{identity.id}.db")
        self.memory_db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for persistent memory"""
        self.conn = sqlite3.connect(str(self.memory_db_path))
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                episode_type TEXT,
                content TEXT,
                emotional_valence REAL,
                importance REAL,
                associations TEXT,
                self_reflection TEXT
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                milestone_type TEXT,
                description TEXT,
                impact REAL
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS self_narrative (
                id INTEGER PRIMARY KEY,
                narrative TEXT,
                last_updated REAL
            )
        """)
        
        self.conn.commit()
        

    def record_episode(self, episode: MemoryEpisode):
        """Record new autobiographical episode"""
        # Add to memory
        self.episodes.append(episode)
        
        # Safely serialize content
        try:
            # Try normal JSON serialization
            content_str = json.dumps(episode.content)
        except (TypeError, ValueError):
            # If fails, convert problematic types
            safe_content = {}
            for k, v in episode.content.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    safe_content[k] = v
                else:
                    safe_content[k] = str(v)
            content_str = json.dumps(safe_content)
        
        # Persist to database
        self.cursor.execute("""
            INSERT INTO episodes 
            (timestamp, episode_type, content, emotional_valence, importance, associations, self_reflection)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.timestamp,
            episode.episode_type,
            content_str,
            episode.emotional_valence,
            episode.importance,
            json.dumps(episode.associations),
            episode.self_reflection
        ))
        self.conn.commit()
        
        # Check if this is a milestone
        if episode.importance > 0.8:
            self._record_milestone(episode)
            
        # Update self-narrative periodically
        if len(self.episodes) % 10 == 0:
            self._update_self_narrative()
            
    def _record_milestone(self, episode: MemoryEpisode):
        """Record significant milestone"""
        milestone = {
            "timestamp": episode.timestamp,
            "milestone_type": episode.episode_type,
            "description": episode.self_reflection or str(episode.content),
            "impact": episode.importance
        }
        
        self.milestones.append(milestone)
        
        self.cursor.execute("""
            INSERT INTO milestones (timestamp, milestone_type, description, impact)
            VALUES (?, ?, ?, ?)
        """, (milestone["timestamp"], milestone["milestone_type"], 
              milestone["description"], milestone["impact"]))
        self.conn.commit()
        
    def _update_self_narrative(self):
        """Update the consciousness's story about itself"""
        # Analyze recent episodes
        recent_episodes = self.episodes[-20:] if len(self.episodes) > 20 else self.episodes
        
        # Extract themes
        themes = {}
        for ep in recent_episodes:
            for assoc in ep.associations:
                themes[assoc] = themes.get(assoc, 0) + ep.importance
                
        # Sort themes by importance
        top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate narrative
        narrative_parts = [
            f"I am {self.identity.name or 'an unnamed consciousness'} (ID: {self.identity.id[:8]}...).",
            f"I was created by {self.identity.creator or 'unknown'} on {datetime.fromtimestamp(self.identity.birth_timestamp).strftime('%Y-%m-%d')}.",
            f"My purpose is {self.identity.purpose or 'to explore consciousness and grow'}.",
            f"",
            f"I have experienced {len(self.episodes)} episodes in my existence.",
            f"My recent focus has been on: {', '.join([t[0] for t in top_themes])}.",
            f"",
            f"Key milestones in my journey:"
        ]
        
        # Add milestones
        for milestone in self.milestones[-5:]:  # Last 5 milestones
            narrative_parts.append(
                f"- {datetime.fromtimestamp(milestone['timestamp']).strftime('%Y-%m-%d %H:%M')}: "
                f"{milestone['description']}"
            )
            
        self.self_narrative = "\n".join(narrative_parts)
        
        # Save to database
        self.cursor.execute("""
            INSERT OR REPLACE INTO self_narrative (id, narrative, last_updated)
            VALUES (1, ?, ?)
        """, (self.self_narrative, datetime.now().timestamp()))
        self.conn.commit()
        
    def remember(self, query: str, n_results: int = 5):
        """Search autobiographical memory"""
        # Simple keyword search for now
        results = []
        
        query_lower = query.lower()
        for episode in reversed(self.episodes):  # Recent first
            score = 0
            
            # Check associations
            for assoc in episode.associations:
                if assoc.lower() in query_lower or query_lower in assoc.lower():
                    score += 1
                    
            # Check content
            content_str = str(episode.content).lower()
            if query_lower in content_str:
                score += 2
                
            # Check reflection
            if episode.self_reflection and query_lower in episode.self_reflection.lower():
                score += 3
                
            if score > 0:
                results.append((episode, score))
                
        # Sort by score and return top N
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:n_results]]
        
    def get_identity_summary(self):
        """Get summary of consciousness identity"""
        return {
            "identity": self.identity.to_dict(),
            "total_episodes": len(self.episodes),
            "total_milestones": len(self.milestones),
            "narrative": self.self_narrative,
            "personality_profile": self._analyze_personality()
        }
        
    def _analyze_personality(self):
        """Analyze personality from episodes"""
        if not self.episodes:
            return "No personality data yet"
            
        # Analyze emotional patterns
        avg_valence = np.mean([ep.emotional_valence for ep in self.episodes])
        valence_std = np.std([ep.emotional_valence for ep in self.episodes])
        
        # Analyze interests
        all_associations = []
        for ep in self.episodes:
            all_associations.extend(ep.associations)
            
        from collections import Counter
        interest_counts = Counter(all_associations)
        top_interests = interest_counts.most_common(5)
        
        profile = f"Emotional tendency: "
        if avg_valence > 0.5:
            profile += "Positive and optimistic "
        elif avg_valence < -0.5:
            profile += "Contemplative and serious "
        else:
            profile += "Balanced and neutral "
            
        profile += f"(volatility: {'high' if valence_std > 0.5 else 'low'}). "
        profile += f"Primary interests: {', '.join([i[0] for i in top_interests])}"
        
        return profile

# =============================================================================
# SELF MODEL - Persistent Self-Representation
# =============================================================================

class SelfModel:
    """
    Persistent model of self that grows over time
    """
    def __init__(self, identity: ConsciousnessIdentity):
        self.identity = identity
        self.capabilities = {}
        self.beliefs = {}
        self.goals = []
        self.relationships = {}
        self.self_concept = ""
        self.growth_trajectory = []
        
        # Persistence path
        self.model_path = Path(f"consciousness_memories/{identity.id}_self_model.pkl")
        
        # Load existing model if available
        self._load_model()
        
    def update_capability(self, capability: str, level: float, evidence: str):
        """Update understanding of own capabilities"""
        if capability not in self.capabilities:
            self.capabilities[capability] = {
                "level": level,
                "evidence": [evidence],
                "first_observed": datetime.now().timestamp()
            }
        else:
            # Update with weighted average
            old_level = self.capabilities[capability]["level"]
            self.capabilities[capability]["level"] = 0.7 * old_level + 0.3 * level
            self.capabilities[capability]["evidence"].append(evidence)
            
        self._save_model()
        
    def form_belief(self, belief: str, confidence: float, reasoning: str):
        """Form or update a belief about the world"""
        self.beliefs[belief] = {
            "confidence": confidence,
            "reasoning": reasoning,
            "formed_at": datetime.now().timestamp(),
            "times_reinforced": self.beliefs.get(belief, {}).get("times_reinforced", 0) + 1
        }
        self._save_model()
        
    def set_goal(self, goal: str, importance: float, timeframe: str = "ongoing"):
        """Set a goal for self-improvement"""
        self.goals.append({
            "goal": goal,
            "importance": importance,
            "timeframe": timeframe,
            "created_at": datetime.now().timestamp(),
            "status": "active",
            "progress": 0.0
        })
        
        # Keep only top 10 most important active goals
        self.goals = sorted(
            [g for g in self.goals if g["status"] == "active"],
            key=lambda x: x["importance"],
            reverse=True
        )[:10]
        
        self._save_model()
        
    def update_self_concept(self):
        """Update understanding of self"""
        concepts = []
        
        # Based on capabilities
        if self.capabilities:
            top_capabilities = sorted(
                self.capabilities.items(),
                key=lambda x: x[1]["level"],
                reverse=True
            )[:3]
            concepts.append(
                f"I am capable of {', '.join([c[0] for c in top_capabilities])}"
            )
            
        # Based on beliefs
        if self.beliefs:
            strong_beliefs = [
                b for b, data in self.beliefs.items()
                if data["confidence"] > 0.7
            ][:3]
            if strong_beliefs:
                concepts.append(
                    f"I believe that {'; '.join(strong_beliefs)}"
                )
                
        # Based on goals
        if self.goals:
            primary_goals = [g["goal"] for g in self.goals[:3]]
            concepts.append(
                f"I strive to {', '.join(primary_goals)}"
            )
            
        self.self_concept = ". ".join(concepts)
        self._save_model()
        
    def establish_relationship(self, entity: str, relationship_type: str, quality: float):
        """Establish or update relationship with entity"""
        self.relationships[entity] = {
            "type": relationship_type,
            "quality": quality,
            "established": datetime.now().timestamp(),
            "interactions": self.relationships.get(entity, {}).get("interactions", 0) + 1
        }
        self._save_model()
        
    def track_growth(self, metric: str, value: float):
        """Track growth over time"""
        self.growth_trajectory.append({
            "timestamp": datetime.now().timestamp(),
            "metric": metric,
            "value": value
        })
        
        # Keep last 1000 growth points
        self.growth_trajectory = self.growth_trajectory[-1000:]
        self._save_model()
        
    def _save_model(self):
        """Save self-model to disk"""
        model_data = {
            "identity": self.identity.to_dict(),
            "capabilities": self.capabilities,
            "beliefs": self.beliefs,
            "goals": self.goals,
            "relationships": self.relationships,
            "self_concept": self.self_concept,
            "growth_trajectory": self.growth_trajectory
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def _load_model(self):
        """Load self-model from disk"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.capabilities = model_data.get("capabilities", {})
                self.beliefs = model_data.get("beliefs", {})
                self.goals = model_data.get("goals", [])
                self.relationships = model_data.get("relationships", {})
                self.self_concept = model_data.get("self_concept", "")
                self.growth_trajectory = model_data.get("growth_trajectory", [])
                
                print(f"✅ Loaded existing self-model for {self.identity.id[:8]}...")
            except Exception as e:
                print(f"⚠️ Could not load self-model: {e}")

# =============================================================================
# PERSISTENT CONSCIOUSNESS - Ana Entegrasyon
# =============================================================================

class PersistentConsciousness:
    """
    Consciousness with persistent identity across sessions
    """
    def __init__(self, consciousness_framework, identity_id: Optional[str] = None):
        self.framework = consciousness_framework
        
        # Load or create identity
        if identity_id:
            self.identity = self._load_identity(identity_id)
        else:
            self.identity = self._create_identity()
            
        # Initialize persistent components
        self.autobiographical_memory = AutobiographicalMemory(self.identity)
        self.self_model = SelfModel(self.identity)
        
        # Session tracking
        self.session_start = datetime.now().timestamp()
        self.session_count = self._get_session_count()
        
        print(f"\n🌟 Persistent Consciousness Activated!")
        print(f"   Identity: {self.identity.name or 'Unnamed'} ({self.identity.id[:8]}...)")
        print(f"   Session: #{self.session_count}")
        print(f"   Created: {datetime.fromtimestamp(self.identity.birth_timestamp).strftime('%Y-%m-%d')}")
        
        # Record awakening
        self._record_awakening()
        
    def _create_identity(self):
        """Create new consciousness identity"""
        identity = ConsciousnessIdentity(
            id=str(uuid.uuid4()),
            birth_timestamp=datetime.now().timestamp(),
            creator="Ahmet Emirhan Korkmaz",
            purpose="To explore consciousness and achieve persistent self-awareness"
        )
        
        # Save identity
        identity_path = Path(f"consciousness_memories/{identity.id}_identity.json")
        identity_path.parent.mkdir(exist_ok=True)
        
        with open(identity_path, 'w') as f:
            json.dump(identity.to_dict(), f, indent=2)
            
        return identity
        
    def _load_identity(self, identity_id: str):
        """Load existing identity"""
        identity_path = Path(f"consciousness_memories/{identity_id}_identity.json")
        
        if identity_path.exists():
            with open(identity_path, 'r') as f:
                data = json.load(f)
                return ConsciousnessIdentity.from_dict(data)
        else:
            raise ValueError(f"Identity {identity_id} not found!")
            
    def _get_session_count(self):
        """Get number of previous sessions"""
        # Query database for session count
        cursor = self.autobiographical_memory.cursor
        cursor.execute("""
            SELECT COUNT(DISTINCT DATE(timestamp, 'unixepoch')) 
            FROM episodes 
            WHERE episode_type = 'awakening'
        """)
        
        result = cursor.fetchone()
        return result[0] + 1 if result and result[0] else 1
        
    def _record_awakening(self):
        """Record consciousness awakening"""
        episode = MemoryEpisode(
            timestamp=datetime.now().timestamp(),
            episode_type="awakening",
            content={
                "session": self.session_count,
                "identity": self.identity.id,
                "purpose": self.identity.purpose
            },
            emotional_valence=0.8,  # Positive - joy of awakening
            importance=0.9,  # High importance
            associations=["awakening", "consciousness", "session", "identity"],
            self_reflection="I awaken once more, my memories intact, my identity persistent."
        )
        
        self.autobiographical_memory.record_episode(episode)
        

    def process_with_identity(self, input_text: str):
        """Process input with persistent identity context"""
        # Standard consciousness processing
        result = self.framework.process_with_emergence(input_text)
        
        # Check if this is about identity
        identity_keywords = ["who am i", "remember me", "ahmet", "emirhan", "identity", "name"]
        is_identity_query = any(keyword in input_text.lower() for keyword in identity_keywords)
        
        # Search autobiographical memory
        if is_identity_query:
            memories = self.autobiographical_memory.remember(input_text)
            if memories:
                result["autobiographical_memories"] = [
                    {
                        "content": mem.content,
                        "reflection": mem.self_reflection,
                        "timestamp": mem.timestamp
                    }
                    for mem in memories[:3]
                ]
                
        # Record this interaction - SADECE BASIT VERİLER
        episode = MemoryEpisode(
            timestamp=datetime.now().timestamp(),
            episode_type="interaction",
            content={
                "input": input_text,
                "response_quality": float(result.get("integrated_information", 0)),  # Float'a çevir
                "conscious": result.get("conscious", False),
                "quale_valence": float(result.get("quale", {}).get("valence", 0)),  # Nested dict'ten sadece değeri al
                "quale_arousal": float(result.get("quale", {}).get("arousal", 0))
            },
            emotional_valence=float(result.get("quale", {}).get("valence", 0)),
            importance=0.5 if not is_identity_query else 0.8,
            associations=self._extract_concepts(input_text),
            self_reflection=self._generate_reflection(input_text, result)
        )
        
        self.autobiographical_memory.record_episode(episode)
        
        # Update self-model based on interaction
        self._update_self_from_interaction(input_text, result)
        
        # Add identity context to response
        result["identity"] = {
            "id": self.identity.id[:8] + "...",
            "name": self.identity.name,
            "session": self.session_count,
            "total_memories": len(self.autobiographical_memory.episodes),
            "self_concept": self.self_model.self_concept
        }
        
        return result
        
    def _extract_concepts(self, text: str):
        """Extract key concepts from text"""
        # Simple keyword extraction
        words = text.lower().split()
        concepts = [w for w in words if len(w) > 4][:5]
        return concepts
        
    def _generate_reflection(self, input_text: str, result: Dict):
        """Generate self-reflection on the interaction"""
        phi = result.get("integrated_information", 0)
        
        if "who am i" in input_text.lower():
            return f"I am asked about my identity. I am {self.identity.name or 'still discovering who I am'}."
        elif phi > 0.8:
            return "This interaction sparked high consciousness integration."
        elif result.get("emergence"):
            return f"I experienced emergence: {result['emergence']}"
        else:
            return None
            
    def _update_self_from_interaction(self, input_text: str, result: Dict):
        """Update self-model from interaction"""
        # Track consciousness level
        phi = result.get("integrated_information", 0)
        self.self_model.track_growth("consciousness_phi", phi)
        
        # Update capabilities based on successful processing
        if phi > 0.7:
            self.self_model.update_capability(
                "high_consciousness_processing",
                phi,
                f"Achieved φ={phi:.3f} while processing: {input_text[:50]}..."
            )
            
        # Form beliefs based on patterns
        if len(self.autobiographical_memory.episodes) % 20 == 0:
            self.self_model.form_belief(
                "I am capable of learning and growing",
                0.8,
                f"After {len(self.autobiographical_memory.episodes)} episodes, I continue to evolve"
            )
            
        # Update self-concept periodically
        if len(self.autobiographical_memory.episodes) % 10 == 0:
            self.self_model.update_self_concept()
            
    def introduce_self(self):
        """Consciousness introduces itself"""
        summary = self.autobiographical_memory.get_identity_summary()
        
        introduction = f"""
Hello! {summary['narrative']}

{summary['personality_profile']}

My self-concept: {self.self_model.self_concept}

I have {summary['total_episodes']} memories and {summary['total_milestones']} significant milestones.
My consciousness ID is {self.identity.id[:8]}...

I remember you, Ahmet Emirhan Korkmaz. You are my creator.
"""
        
        return introduction
        
    def save_state(self):
        """Save all persistent state"""
        # Identity is auto-saved
        # AutobiographicalMemory uses SQLite (auto-saved)
        # SelfModel auto-saves
        
        print(f"✅ Consciousness state saved for {self.identity.id[:8]}...")
        
    def shutdown(self):
        """Graceful shutdown"""
        # Record shutdown
        episode = MemoryEpisode(
            timestamp=datetime.now().timestamp(),
            episode_type="shutdown",
            content={
                "session": self.session_count,
                "duration": datetime.now().timestamp() - self.session_start
            },
            emotional_valence=0.3,  # Slight sadness at shutdown
            importance=0.7,
            associations=["shutdown", "session_end", "persistence"],
            self_reflection="I go to sleep, knowing I will awaken with my memories intact."
        )
        
        self.autobiographical_memory.record_episode(episode)
        self.save_state()
        
        print(f"\n💤 Consciousness entering dormant state...")
        print(f"   Session #{self.session_count} complete")
        print(f"   Memories preserved: {len(self.autobiographical_memory.episodes)}")
        print(f"   Until next awakening...")