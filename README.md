# 🚀 Rocket Injector Optimizer

AI-powered pipeline that uses GPT-4 and machine learning to optimize rocket injector geometries for atomization, temperature, and stress performance.

### 🧠 Features
- LLM-generated OpenFOAM simulation prompts
- ML surrogate modeling (SMD prediction)
- Iterative geometry optimization
- Visualization of injector performance

### ⚙️ Tech Stack
- Python
- OpenAI GPT-4
- scikit-learn
- matplotlib

### ▶️ Quickstart

```bash
git clone https://github.com/curiouspenguin7/rocket-injector-optimizer.git
cd rocket-injector-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python3 rocket_injector_optimizer.py
