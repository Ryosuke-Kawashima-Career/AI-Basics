from agents.tools import format_search_results, tag_citations

def main():
    results = [{"title": "RAG vs fine-tuning", "url": "https://www.actian.com/blog/databases/should-you-use-rag-or-fine-tune-your-llm/", "snippet": "..."}]
    print(format_search_results(results))
    print(tag_citations("RAG is generally cheaper than fine-tuning.", results))

if __name__ == "__main__":
    main()

          