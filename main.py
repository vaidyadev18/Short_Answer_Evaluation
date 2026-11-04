import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

questions = {
    1: {
        "question": "What is Machine Learning?",
        "answer": "Machine Learning is a branch of artificial intelligence that allows computers to learn from data and improve their performance without explicit programming."
    },
    2: {
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence is a field of computer science that enables machines to perform tasks that normally require human intelligence."
    },
    3: {
        "question": "What is Deep Learning?",
        "answer": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns from data."
    },
    4: {
        "question": "What is NLP?",
        "answer": "Natural Language Processing is a branch of artificial intelligence that helps computers understand, process and generate human language."
    },
    5: {
        "question": "What is Data Science?",
        "answer": "Data Science is the process of collecting, analyzing and interpreting data using statistics, programming and machine learning."
    }
}


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def evaluate_answer(student_answer, reference_answer, max_marks=10):

    student = preprocess(student_answer)
    reference = preprocess(reference_answer)

    if not student.strip() or not reference.strip():
        return 0, "No answer provided"

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([student, reference])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    score = similarity * max_marks

    if similarity >= 0.70:
        feedback = "Excellent Answer"
    elif similarity >= 0.50:
        feedback = "Good Answer"
    elif similarity >= 0.30:
        feedback = "Average Answer"
    else:
        feedback = "Needs Improvement"

    return round(score, 2), feedback


def main():

    print("\n========================================")
    print(" AUTOMATIC SHORT ANSWER EVALUATION")
    print("========================================")

    total_score = 0
    total_questions = len(questions)

    for number, data in questions.items():

        print("\nQuestion", number)
        print(data["question"])

        student_answer = input("\nEnter your answer: ")

        marks, feedback = evaluate_answer(
            student_answer,
            data["answer"],
            10
        )

        total_score += marks

        print("\nReference Answer:")
        print(data["answer"])

        print("\nMarks Obtained:", marks, "/10")
        print("Feedback:", feedback)

        print("----------------------------------------")

    print("\n========== FINAL RESULT ==========")
    print("Total Marks:", round(total_score, 2), "/", total_questions * 10)

    percentage = (total_score / (total_questions * 10)) * 100

    print("Percentage:", round(percentage, 2), "%")

    print("==================================")


if __name__ == "__main__":
    main()
