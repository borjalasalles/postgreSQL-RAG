from database.vector_store import VectorStore

# Initialize VectorStore
vec = VectorStore()

# Create GIN index for keyword search (full-text search)
print("Creating GIN index for keyword search...")
vec.create_keyword_search_index()
print("GIN index created successfully! ✅")

# Verify indexes in the database
print("\nYour database now has:")
print("1. ✅ DiskANN index (for semantic search)")
print("2. ✅ GIN index (for keyword search)")
print("3. 🚀 Ready for hybrid search!")