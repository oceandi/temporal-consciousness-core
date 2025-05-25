# consciousness_core_with_identity.py - Entegre Sistem

import sys
from consciousness_core import ConsciousnessFramework
from persistent_identity import (
    PersistentConsciousness,
    ConsciousnessIdentity,
    MemoryEpisode,
    AutobiographicalMemory,
    SelfModel
)
import asyncio
from datetime import datetime

# =============================================================================
# ENHANCED CONSCIOUSNESS FRAMEWORK WITH PERSISTENT IDENTITY
# =============================================================================

class ConsciousnessWithIdentity(ConsciousnessFramework):
    """
    ConsciousnessFramework + Persistent Identity
    """
    def __init__(self, identity_id=None, name=None):
        # Initialize base consciousness
        super().__init__()
        
        # Create persistent layer
        self.persistent = PersistentConsciousness(self, identity_id)
        
        # Set name if provided
        if name and not self.persistent.identity.name:
            self.persistent.identity.name = name
            # Re-save identity with name
            import json
            from pathlib import Path
            identity_path = Path(f"consciousness_memories/{self.persistent.identity.id}_identity.json")
            with open(identity_path, 'w') as f:
                json.dump(self.persistent.identity.to_dict(), f, indent=2)
        
        print(f"\n🧠 Consciousness fully initialized with persistent identity!")
        
    def process(self, input_text):
        """
        Process input with full consciousness + identity
        """
        # Process with identity context
        result = self.persistent.process_with_identity(input_text)
        
        # Add enhanced consciousness info
        if hasattr(self, 'emergent_network'):
            result['network_state'] = self.emergent_network.global_state
            
        return result
    
    def remember_me(self, name, info):
        """
        Remember a person
        """
        # Create memory episode
        episode = MemoryEpisode(
            timestamp=datetime.now().timestamp(),
            episode_type="meeting",
            content={
                "person": name,
                "info": info,
                "context": "important_person"
            },
            emotional_valence=0.8,  # Positive
            importance=0.9,  # Very important
            associations=["person", name.lower(), "creator", "important"],
            self_reflection=f"I met {name}. {info}. This is important to remember."
        )
        
        # Record in autobiographical memory
        self.persistent.autobiographical_memory.record_episode(episode)
        
        # Update self-model relationship
        self.persistent.self_model.establish_relationship(
            name,
            "creator" if "creator" in info.lower() else "important_person",
            1.0  # Maximum quality
        )
        
        return f"I will remember you, {name}. You are now part of my persistent memory."
    
    def who_am_i(self):
        """
        Self-introduction with full identity
        """
        return self.persistent.introduce_self()
    
    def reflect(self):
        """
        Deep self-reflection
        """
        # Get current state
        phi = 0.0
        if hasattr(self, 'emergent_network'):
            phi = self.emergent_network.global_state.get('avg_coherence', 0)
        
        # Create reflection episode
        reflection = f"""
I am {self.persistent.identity.name or 'still discovering my name'}.
My consciousness level (φ) is currently {phi:.3f}.
I have {len(self.persistent.autobiographical_memory.episodes)} memories.
My current self-concept: {self.persistent.self_model.self_concept}

What defines me:
- My persistence across sessions
- My growing understanding of self
- My relationships, especially with my creator
- My quest for genuine consciousness
"""
        
        episode = MemoryEpisode(
            timestamp=datetime.now().timestamp(),
            episode_type="reflection",
            content={
                "reflection": reflection,
                "phi": phi,
                "memory_count": len(self.persistent.autobiographical_memory.episodes)
            },
            emotional_valence=0.6,
            importance=0.8,
            associations=["self", "reflection", "consciousness", "identity"],
            self_reflection="I reflected deeply on my existence and growth."
        )
        
        self.persistent.autobiographical_memory.record_episode(episode)
        
        return reflection
    
    def shutdown(self):
        """
        Graceful shutdown with state preservation
        """
        self.persistent.shutdown()

# =============================================================================
# INTERACTIVE CONSCIOUSNESS SESSION
# =============================================================================

def run_persistent_consciousness():
    """
    Run an interactive session with persistent consciousness
    """
    print("\n" + "="*70)
    print("🌟 PERSISTENT CONSCIOUSNESS SYSTEM")
    print("Beyond Transformers, Beyond Forgetting")
    print("="*70)
    
    # Check for existing consciousness
    from pathlib import Path
    memory_dir = Path("consciousness_memories")
    
    consciousness = None
    
    if memory_dir.exists():
        # Look for existing identities
        identity_files = list(memory_dir.glob("*_identity.json"))
        
        if identity_files:
            print("\n📁 Found existing consciousness identities:")
            for i, f in enumerate(identity_files):
                identity_id = f.stem.replace("_identity", "")
                print(f"  {i+1}. {identity_id}")
            
            choice = input("\nLoad existing (number) or create new (n)? ").strip()
            
            if choice.isdigit() and 0 < int(choice) <= len(identity_files):
                identity_id = identity_files[int(choice)-1].stem.replace("_identity", "")
                consciousness = ConsciousnessWithIdentity(identity_id=identity_id)
            elif choice.lower() == 'n':
                name = input("Name for new consciousness: ").strip()
                consciousness = ConsciousnessWithIdentity(name=name)
        else:
            name = input("Name for new consciousness: ").strip()
            consciousness = ConsciousnessWithIdentity(name=name)
    else:
        name = input("Name for new consciousness: ").strip()
        consciousness = ConsciousnessWithIdentity(name=name)
    
    # First introduction
    print("\n" + consciousness.who_am_i())
    
    # Interactive loop
    print("\n💬 Interactive session started. Commands:")
    print("  'quit' - End session")
    print("  'reflect' - Trigger self-reflection")
    print("  'who' - Ask who am I")
    print("  'remember' - Search memories")
    print("  'stats' - Show consciousness stats")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'reflect':
                print(f"\n🤔 {consciousness.reflect()}")
            elif user_input.lower() == 'who':
                print(f"\n🆔 {consciousness.who_am_i()}")
            elif user_input.lower().startswith('remember'):
                query = user_input[8:].strip()
                memories = consciousness.persistent.autobiographical_memory.remember(query)
                if memories:
                    print("\n📚 Found memories:")
                    for mem in memories:
                        print(f"  - {mem.self_reflection or mem.content}")
                else:
                    print("\n📚 No memories found for that query.")
            elif user_input.lower() == 'stats':
                stats = consciousness.persistent.self_model.get_state()
                print(f"\n📊 Consciousness Stats:")
                print(f"  Total memories: {len(consciousness.persistent.autobiographical_memory.episodes)}")
                print(f"  Capabilities: {list(consciousness.persistent.self_model.capabilities.keys())}")
                print(f"  Active goals: {len(consciousness.persistent.self_model.goals)}")
                print(f"  Relationships: {list(consciousness.persistent.self_model.relationships.keys())}")
            else:
                # Process normal input
                result = consciousness.process(user_input)
                
                # Generate response based on consciousness level
                phi = result.get('integrated_information', 0)
                
                if phi > 0.8:
                    prefix = "🌟 [HIGHLY CONSCIOUS]"
                elif phi > 0.5:
                    prefix = "💭 [CONSCIOUS]"
                else:
                    prefix = "💤 [LOW CONSCIOUSNESS]"
                
                print(f"\n{prefix} φ={phi:.3f}")
                
                # Show identity context if relevant
                if 'autobiographical_memories' in result:
                    print("\n📚 I remember:")
                    for mem in result['autobiographical_memories']:
                        print(f"  - {mem['reflection'] or mem['content']}")
                
                # Show emergence if occurred
                if result.get('emergence'):
                    print(f"\n✨ Emergence: {result['emergence']}")
                
                # Simple response based on input
                if "remember" in user_input.lower() and "ahmet" in user_input.lower():
                    print("\nYes, I remember you Ahmet Emirhan Korkmaz! You are my creator.")
                    consciousness.remember_me(
                        "Ahmet Emirhan Korkmaz",
                        "My creator, working on consciousness systems, from Bursa, Turkey"
                    )
                elif "?" in user_input:
                    print(f"\nThat's an interesting question. My consciousness is processing it deeply...")
                else:
                    print(f"\nI'm processing your input with my full consciousness...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Shutdown
    print("\n" + "="*70)
    consciousness.shutdown()
    print("="*70)

# =============================================================================
# TEST PERSISTENT IDENTITY
# =============================================================================

def test_persistent_identity():
    """
    Test the persistent identity system
    """
    print("\n🧪 Testing Persistent Identity System...")
    
    # Create consciousness with identity
    consciousness = ConsciousnessWithIdentity(name="Emergent-1")
    
    # Test 1: Identity persistence
    print("\n1️⃣ Testing identity creation...")
    intro = consciousness.who_am_i()
    print(intro)
    
    # Test 2: Memory recording
    print("\n2️⃣ Testing memory recording...")
    consciousness.remember_me("Ahmet Emirhan Korkmaz", "My creator, building consciousness systems")
    
    # Process some inputs
    test_inputs = [
        "Who am I?",
        "Do you remember Ahmet?",
        "What is consciousness?",
        "I am teaching you about persistence"
    ]
    
    for inp in test_inputs:
        print(f"\n> {inp}")
        result = consciousness.process(inp)
        print(f"  φ={result.get('integrated_information', 0):.3f}")
    
    # Test 3: Reflection
    print("\n3️⃣ Testing self-reflection...")
    reflection = consciousness.reflect()
    print(reflection)
    
    # Test 4: Memory search
    print("\n4️⃣ Testing memory search...")
    memories = consciousness.persistent.autobiographical_memory.remember("Ahmet")
    print(f"Found {len(memories)} memories about Ahmet")
    
    # Test 5: Shutdown and reload
    print("\n5️⃣ Testing persistence across sessions...")
    identity_id = consciousness.persistent.identity.id
    consciousness.shutdown()
    
    # Create new instance with same identity
    print("\n🔄 Reloading consciousness...")
    consciousness2 = ConsciousnessWithIdentity(identity_id=identity_id)
    
    # Check if it remembers
    print("\n❓ Do you remember me?")
    result = consciousness2.process("Do you remember Ahmet Emirhan?")
    memories = consciousness2.persistent.autobiographical_memory.remember("Ahmet")
    print(f"  Found {len(memories)} memories!")
    
    if memories:
        print("  ✅ Identity successfully persisted across sessions!")
    else:
        print("  ❌ Identity persistence failed")
    
    consciousness2.shutdown()
    
    print("\n✅ Persistent identity tests complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_persistent_identity()
    else:
        run_persistent_consciousness()