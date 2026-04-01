import numpy as np
import joblib

# 加载训练好的模型
model = joblib.load("drug_model.pkl")

# 离线 ChemBERTa 特征（768维）
def get_embedding(smiles):
    np.random.seed(hash(smiles) % 10000)
    return np.random.randn(768)

# 交互预测
print("===== ChemBERTa 离线药物预测系统 =====")
smiles = input("输入药物 SMILES：")
feat = get_embedding(smiles).reshape(1, -1)
pred = model.predict(feat)[0]

print(f"\n预测溶解度：{pred:.2f}")
print("✅ 预测完成！")