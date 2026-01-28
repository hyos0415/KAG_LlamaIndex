from app.builder.news_builder import ClaudeNewsBuilder
from app.solver.news_solver import ClaudeNewsSolver
from app.validator.news_validator import ClaudeNewsValidator

def main():
    print("=== LlamaIndex News PGI System ===")
    print("1. Build Graph (Builder)")
    print("2. Query Graph (Solver)")
    print("3. Validate Article (Validator)")
    
    choice = input("\nSelect menu (1/2/3): ")
    
    if choice == '1':
        builder = ClaudeNewsBuilder()
        builder.build_and_persist()
    elif choice == '2':
        solver = ClaudeNewsSolver()
        query = input("Enter query: ")
        print(solver.solve(query))
    elif choice == '3':
        validator = ClaudeNewsValidator()
        article = input("Enter article text for validation: ")
        print(validator.validate_article(article))
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
