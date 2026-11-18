import pymupdf

def main():
    document = pymupdf.open("data/sample.pdf")

    for page in document:
        text = page.get_text()
        print(text)

    document.close()

if __name__ == "__main__":
    main()
