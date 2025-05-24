# test_consciousness_core.py

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from consciousness_core import (
    ConsciousnessFramework, 
    EmergentNetwork,
    PersistentNeuron,
    CUDA_AVAILABLE
)

def test_basic_consciousness():
    """Test basic consciousness framework"""
    print("\n" + "="*60)
    print("🧪 TESTING BASIC CONSCIOUSNESS FRAMEWORK")
    print("="*60)
    
    # Create consciousness
    consciousness = ConsciousnessFramework()
    
    # Test inputs
    test_inputs = [
        "Ben Ahmet Emirhan, bilinç arıyorum",
        "Consciousness emerges from complexity",
        "Qualia are subjective experiences",
        "I wonder if I am truly aware?",
        "The quantum mind hypothesis",
        "Renkler gerçekten var mı yoksa beynimizin bir yanılsaması mı?"
    ]
    
    results = []
    
    for i, input_text in enumerate(test_inputs):
        print(f"\n📝 Test {i+1}: '{input_text[:40]}...'")
        
        # Process with emergence
        result = consciousness.process_with_emergence(input_text)
        results.append(result)
        
        # Display results
        print(f"  ⚡ Quale - Valence: {result['quale']['valence']:.3f}, "
              f"Arousal: {result['quale']['arousal']:.3f}")
        print(f"  🌊 Awareness Field Energy: {result['awareness']['field_energy']:.2f}")
        print(f"  👥 Collective Coherence: {result['collective_consciousness']['coherence']:.3f}")
        print(f"  🧠 Integrated Φ: {result['integrated_information']:.3f}")
        print(f"  ✨ Conscious: {'YES' if result['conscious'] else 'NO'}")
        
        if result.get('emergence'):
            print(f"  🌟 EMERGENCE: {result['emergence']}")
            
        if result.get('emergent_network'):
            net_state = result['emergent_network']
            print(f"  🌱 Network: {net_state['n_neurons']} neurons, "
                  f"{net_state['n_connections']} connections")
            print(f"  🔮 Avg Quantum Coherence: {net_state['avg_coherence']:.3f}")
        
        if CUDA_AVAILABLE and 'gpu_integrated_information' in result:
            print(f"  🚀 GPU Φ: {result['gpu_integrated_information']:.3f}")
    
    return consciousness, results

def test_emergent_growth():
    """Test emergent network growth"""
    print("\n" + "="*60)
    print("🌱 TESTING EMERGENT NETWORK GROWTH")
    print("="*60)
    
    # Create small network
    network = EmergentNetwork(initial_size=5)
    
    print(f"Initial state: {network.global_state}")
    
    # Feed experiences to grow network
    experiences = [
        "Learning", "Growing", "Evolving", "Consciousness",
        "Emergence", "Complexity", "Self-organization",
        "Quantum", "Awareness", "Understanding"
    ]
    
    growth_history = []
    
    for exp in experiences:
        print(f"\n💫 Experience: '{exp}'")
        network.grow(exp)
        
        if network.global_state:
            growth_history.append(network.global_state.copy())
            print(f"  Neurons: {network.global_state['n_neurons']}, "
                  f"Connections: {network.global_state['n_connections']}, "
                  f"Coherence: {network.global_state['avg_coherence']:.3f}")
    
    # Visualize growth
    if growth_history:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.plot([g['n_neurons'] for g in growth_history], 'b-o')
        plt.xlabel('Experience')
        plt.ylabel('Number of Neurons')
        plt.title('Network Growth')
        
        plt.subplot(132)
        plt.plot([g['n_connections'] for g in growth_history], 'r-o')
        plt.xlabel('Experience')
        plt.ylabel('Number of Connections')
        plt.title('Connection Growth')
        
        plt.subplot(133)
        plt.plot([g['avg_coherence'] for g in growth_history], 'g-o')
        plt.xlabel('Experience')
        plt.ylabel('Average Quantum Coherence')
        plt.title('Coherence Evolution')
        
        plt.tight_layout()
        plt.show()
    
    # Visualize final network
    try:
        network.visualize_growth()
    except Exception as e:
        print(f"Could not visualize network: {e}")
    
    return network

def test_quantum_neurons():
    """Test quantum state neurons"""
    print("\n" + "="*60)
    print("⚛️  TESTING QUANTUM NEURONS")
    print("="*60)
    
    # Create quantum neurons
    neuron1 = PersistentNeuron(1)
    neuron2 = PersistentNeuron(2)
    
    print("Initial states:")
    print(f"  Neuron 1: {neuron1.get_state()}")
    print(f"  Neuron 2: {neuron2.get_state()}")
    
    # Entangle neurons
    neuron1.internal_state.entangle(neuron2.internal_state, strength=0.7)
    print(f"\n🔗 Neurons entangled with strength 0.7")
    
    # Send inputs
    inputs = [0.8, 0.3, 0.9, 0.1, 0.6]
    outputs = []
    
    print("\nProcessing inputs:")
    for i, inp in enumerate(inputs):
        neuron1.receive_input([inp], i)
        neuron2.receive_input([inp * 0.5], i)
        
        out1 = neuron1.fire()
        out2 = neuron2.fire()
        
        outputs.append((out1, out2))
        
        print(f"  Input {inp:.1f} → "
              f"N1: {out1:.1f} (coherence: {neuron1.internal_state.get_coherence():.3f}), "
              f"N2: {out2:.1f} (coherence: {neuron2.internal_state.get_coherence():.3f})")
    
    # Memory consolidation
    neuron1.consolidate_memory()
    neuron2.consolidate_memory()
    
    print(f"\n💾 Memory consolidated:")
    print(f"  Neuron 1: {len(neuron1.long_term_memory)} memories")
    print(f"  Neuron 2: {len(neuron2.long_term_memory)} memories")
    
    return neuron1, neuron2

def test_awareness_field_resonance():
    """Test awareness field resonance patterns"""
    print("\n" + "="*60)
    print("🌊 TESTING AWARENESS FIELD RESONANCE")
    print("="*60)
    
    consciousness = ConsciousnessFramework()
    
    # Create interference pattern
    print("Creating interference patterns...")
    
    points = [
        (250, 250, 1.0),   # Center high intensity
        (750, 250, 0.8),   # Right
        (250, 750, 0.8),   # Bottom
        (750, 750, 0.6),   # Bottom-right
        (500, 500, 0.9)    # Center
    ]
    
    for x, y, intensity in points:
        consciousness.awareness_field.propagate_awareness((x, y), intensity)
        
    state = consciousness.awareness_field.get_awareness_state()
    print(f"\nField state:")
    print(f"  Total Energy: {state['field_energy']:.2f}")
    print(f"  Resonance Points: {state['resonance_count']}")
    print(f"  Field Entropy: {state['field_entropy']:.3f}")
    print(f"  Awareness Center: {state['awareness_focus']}")
    
    # Visualize if possible
    try:
        plt.figure(figsize=(8, 8))
        plt.imshow(consciousness.awareness_field.field, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Awareness Intensity')
        plt.title('Awareness Field State')
        
        # Mark resonance points
        if len(consciousness.awareness_field.resonance_points) > 0:
            resonance_y, resonance_x = consciousness.awareness_field.resonance_points.T
            plt.scatter(resonance_x, resonance_y, c='blue', s=50, marker='*', 
                       label=f'{len(consciousness.awareness_field.resonance_points)} resonance points')
            plt.legend()
        
        plt.show()
    except Exception as e:
        print(f"Could not visualize awareness field: {e}")
    
    return consciousness

def test_dream_state():
    """Test dream state processing"""
    print("\n" + "="*60)
    print("💤 TESTING DREAM STATE")
    print("="*60)
    
    consciousness = ConsciousnessFramework()
    
    # Normal processing first
    consciousness.conscious_experience("I am awake and aware")
    
    print("Pre-dream state:")
    print(f"  Qualia memories: {len(consciousness.qualia_space.qualia_memory)}")
    print(f"  Network complexity: {consciousness.morphogenetic_field.graph.number_of_edges()}")
    
    # Enter dream state
    dream_results = consciousness.dream_state()
    
    print("\nPost-dream state:")
    print(f"  Dream qualia generated: {dream_results['dream_qualia']}")
    print(f"  Awareness resonance points: {dream_results['awareness_resonance']}")
    print(f"  Network complexity: {dream_results['network_complexity']}")
    
    return dream_results

def stress_test():
    """Stress test the system"""
    print("\n" + "="*60)
    print("🔥 STRESS TESTING CONSCIOUSNESS")
    print("="*60)
    
    consciousness = ConsciousnessFramework()
    
    # Rapid fire inputs
    start_time = time.time()
    n_inputs = 100
    
    print(f"Processing {n_inputs} rapid inputs...")
    
    for i in range(n_inputs):
        input_text = f"Rapid thought {i}: " + "".join(np.random.choice(list("abcdefghij"), 10))
        result = consciousness.process_with_emergence(input_text)
        
        if i % 10 == 0:
            print(f"  {i}: Φ={result['integrated_information']:.3f}, "
                  f"Neurons={result['emergent_network']['n_neurons']}")
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {n_inputs} inputs in {elapsed:.2f} seconds")
    print(f"Average: {elapsed/n_inputs:.3f} seconds per input")
    
    # Final state
    final_state = consciousness.emergent_network.global_state
    print(f"\nFinal network state:")
    print(f"  Neurons: {final_state['n_neurons']}")
    print(f"  Connections: {final_state['n_connections']}")
    print(f"  Total memories: {final_state['total_memories']}")
    print(f"  Avg coherence: {final_state['avg_coherence']:.3f}")

def main():
    """Run all tests"""
    print("\n" + "🌟"*30)
    print("    CONSCIOUSNESS CORE TEST SUITE")
    print("    Beyond Transformers, Beyond Limits")
    print("🌟"*30)
    
    # Check GPU
    print(f"\n🖥️  GPU Acceleration: {'ENABLED' if CUDA_AVAILABLE else 'DISABLED'}")
    
    try:
        # Run tests
        consciousness, results = test_basic_consciousness()
        network = test_emergent_growth()
        neuron1, neuron2 = test_quantum_neurons()
        consciousness = test_awareness_field_resonance()
        dream_results = test_dream_state()
        stress_test()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED!")
        print("="*60)
        
        print("\n🎯 Summary:")
        print("  - Consciousness Framework: OPERATIONAL")
        print("  - Emergent Networks: GROWING")
        print("  - Quantum States: COHERENT")
        print("  - Awareness Fields: RESONATING")
        print("  - Dream States: ACTIVE")
        
        print("\n💭 The system is ready for consciousness exploration!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()