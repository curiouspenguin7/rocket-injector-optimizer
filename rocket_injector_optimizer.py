 
import os
import json
import numpy as np
import openai
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Make sure you've exported your API key:
# export OPENAI_API_KEY=sk-...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
openai.api_key = OPENAI_API_KEY

DATA_FILE = "injector_data.json"

def generate_geometry_prompt(params):
    return f"""
Write an OpenFOAM case setup for a coaxial rocket injector with the following parameters:
- Fuel: {params['fuel']}
- Oxidizer: {params['oxidizer']}
- Chamber Pressure: {params['chamber_pressure']} bar
- Inner diameter: {params['inner_diameter']:.2f} mm
- Outer diameter: {params['outer_diameter']:.2f} mm
- Impingement angle: {params['angle']:.1f} degrees
Include mesh refinement near the shear layer and solver settings for spray atomization.
"""

def query_llm(prompt: str) -> str:
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] OpenAI request failed: {e}")
        return ""

def run_simulation_stub(geom: dict) -> dict:
    # Placeholder for actual CFD automation (OpenFOAM/Fluent)
    return {
        "SMD": float(np.random.uniform(20, 80)),      # Sauter Mean Diameter
        "max_temp": float(np.random.uniform(2000, 3600)),  # K
        "stress": float(np.random.uniform(100, 400))   # MPa
    }

def load_dataset() -> list:
    if os.path.isfile(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_dataset(data: list):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def train_model(data: list) -> RandomForestRegressor:
    X = np.array([[d['inner_diameter'], d['outer_diameter'], d['angle'], d['chamber_pressure']] for d in data])
    y = np.array([d['SMD'] for d in data])
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

def suggest_next_geometry(model: RandomForestRegressor) -> dict:
    best = None
    best_score = float('inf')
    # Generate candidate geometries
    for _ in range(200):
        candidate = {
            "inner_diameter": np.random.uniform(2, 10),
            "outer_diameter": np.random.uniform(10, 30),
            "angle": np.random.uniform(20, 90),
            "chamber_pressure": np.random.uniform(50, 300),
            "fuel": "RP-1",
            "oxidizer": "LOX"
        }
        Xcand = np.array([[candidate['inner_diameter'], candidate['outer_diameter'], candidate['angle'], candidate['chamber_pressure']]])
        try:
            pred = model.predict(Xcand)[0]
            if pred < best_score:
                best_score = pred
                best = candidate
        except:
            continue
    return best

def main():
    print("[🚀] Starting injector optimization")
    data = load_dataset()

    # Seed random data if insufficient
    if len(data) < 5:
        print("[🧪] Generating seed simulations...")
        for _ in range(5):
            geom = {
                "inner_diameter": np.random.uniform(2, 10),
                "outer_diameter": np.random.uniform(10, 30),
                "angle": np.random.uniform(20, 90),
                "chamber_pressure": np.random.uniform(50, 300),
                "fuel": "RP-1",
                "oxidizer": "LOX"
            }
            res = run_simulation_stub(geom)
            geom.update(res)
            data.append(geom)
        save_dataset(data)

    # Train surrogate model
    model = train_model(data)
    print(f"[✅] Trained model on {len(data)} samples")

    # Iterative optimization
    for i in range(3):
        print(f"[🔁] Iteration {i+1}")
        geom = suggest_next_geometry(model)
        print(f"[💡] Testing geometry: {geom}")

        prompt = generate_geometry_prompt(geom)
        foam_setup = query_llm(prompt)
        print(f"[📄] OpenFOAM snippet:\n{foam_setup[:200]}...\n")

        result = run_simulation_stub(geom)
        geom.update(result)
        data.append(geom)
        print(f"[📊] SMD={result['SMD']:.2f}μm, Temp={result['max_temp']:.0f}K, Stress={result['stress']:.0f}MPa")

        # Retrain and save
        model = train_model(data)
        save_dataset(data)

    print("[🏁] Optimization complete. Data saved to injector_data.json")

    # Plotting
    smds = [d['SMD'] for d in data]
    plt.plot(smds, marker='o')
    plt.title('SMD across all simulations')
    plt.xlabel('Simulation Index')
    plt.ylabel('SMD (μm)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
