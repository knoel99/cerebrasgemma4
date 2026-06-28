from cerebrasgemma4 import stream

if __name__ == "__main__":
    for token in stream([{"role": "user", "content": "Say hello in one short sentence."}]):
        print(token, end="", flush=True)
    print()