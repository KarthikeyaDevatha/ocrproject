import sys
from inference.hybrid_pipeline import HybridPipeline

print("=== Mathpix Mode ===")
pipe = HybridPipeline(mode="mathpix", verbose=False)
res = pipe.process("samples/page.png")
print("TEXT:")
print(res.text)
print("MATHPIX RESULT:")
print(res.mathpix_result)
print()

print("=== TrOCR Base Mode ===")
pipe2 = HybridPipeline(mode="trocr", trocr_model="base", verbose=False)
pipe2.postprocessor.use_spellcheck = False # disable spellcheck to see raw
res2 = pipe2.process("samples/page.png")
print("TEXT:")
print(res2.text)
print()

print("=== TrOCR Large Mode ===")
pipe3 = HybridPipeline(mode="trocr", trocr_model="large", verbose=False)
pipe3.postprocessor.use_spellcheck = False # disable spellcheck to see raw
res3 = pipe3.process("samples/page.png")
print("TEXT:")
print(res3.text)
