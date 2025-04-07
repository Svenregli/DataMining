from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-base")

test_embedding = model.encode(["Represent the abstract: How do AI agents affect organizations?"])[0]

print("✅ Embedding generated. First 5 values:", test_embedding[:5])
