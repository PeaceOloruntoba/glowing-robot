import pandas as pd
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
import sys
import json # Import json to output structured data

# NLTK Downloads (ensure these are run once, or uncomment if running for the first time)
for corpus in ['movie_reviews', 'punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{corpus}' if corpus in ['movie_reviews', 'stopwords', 'wordnet'] else f'tokenizers/{corpus}')
    except nltk.downloader.DownloadError:
        nltk.download(corpus)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def infer_university_type(text):
    if pd.isna(text):
        return 'unknown'
    text = text.lower()
    if 'private' in text or 'aul' in text or 'covenant' in text or 'babcock' in text:
        return 'private'
    if 'public' in text or 'unilag' in text or 'oau' in text or 'unn' in text or 'ui' in text or 'futo' in text or 'lasu' in text or 'uniben' in text or 'unilorin' in text:
        return 'public'
    return 'unknown'

def prepare_training_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    data = []
    for doc, category in documents:
        text = ' '.join(doc)
        label = 'positive' if category == 'pos' else 'negative'
        data.append({'text': text, 'sentiment': label})
    return pd.DataFrame(data)

# Sample data for fallback if no CSV is provided
sample_data = pd.DataFrame({
    'created_at': [
        '2025-05-01 08:15:23', '2025-05-01 12:30:45', '2025-05-02 09:22:10', '2025-05-02 14:50:33',
        '2025-05-03 10:05:12', '2025-05-03 16:20:55', '2025-05-04 07:45:30', '2025-05-04 13:10:15',
        '2025-05-05 11:25:40', '2025-05-05 18:40:22', '2025-05-06 09:00:50', '2025-05-06 15:30:12',
        '2025-05-07 10:15:33', '2025-05-07 17:20:44', '2025-05-08 08:55:21', '2025-05-08 14:10:30',
        '2025-05-09 11:45:12', '2025-05-09 16:25:50', '2025-05-10 09:30:22', '2025-05-10 15:50:33',
        '2025-05-11 10:05:44', '2025-05-11 17:20:15', '2025-05-12 08:45:22', '2025-05-12 14:10:30',
        '2025-05-13 09:25:41', '2025-05-13 16:30:12', '2025-05-14 10:00:50', '2025-05-14 15:45:22',
        '2025-05-15 08:30:33', '2025-05-15 14:50:44', '2025-05-16 09:15:21', '2025-05-16 16:20:30',
        '2025-05-17 10:05:12', '2025-05-17 15:30:50', '2025-05-18 08:45:22', '2025-05-18 14:10:33',
        '2025-05-19 09:25:44', '2025-05-19 16:30:12', '2025-05-20 10:00:50', '2025-05-20 15:45:22',
        '2025-05-21 08:30:33', '2025-05-21 14:50:44', '2025-05-22 09:15:21', '2025-05-22 16:20:30',
        '2025-05-23 10:05:12', '2025-05-23 15:30:50', '2025-05-24 08:45:22', '2025-05-24 14:10:33',
        '2025-05-25 09:25:44', '2025-05-25 16:30:12'
    ],
    'text': [
        'The new lecture halls at Unilag are top-notch! Spacious, well-lit, and those projectors actually work! #UnilagVibes',
        'Why is the Wi-Fi at OAU always down? Can’t even access the e-learning platform for assignments. Frustrating! #OAUstruggles',
        'Just moved into AUL hostel. Modern facilities, clean bathrooms, but the internet is spotty. Fix it, please! #AULife',
        'The cafeteria at UI is a lifesaver. Affordable food and open late. Wish they had more vegan options though. #UIstudent',
        'Labs at Covenant Uni are world-class! Modern equipment and accessible. Makes studying CS so much easier. #CovenantPride',
        'UNN restrooms are a mess. Always dirty and half the taps don’t work. Admin needs to step up! #UNNissues',
        'New hostel at ABU is amazing! Spacious rooms and Wi-Fi everywhere. Finally feels like home. #ABUlife',
        'Transportation at FUTO is a nightmare. Buses never on time and overcrowded. Fix this! #FUTOprobs',
        'The library at Babcock has every book I need plus e-platforms. Studying here is a breeze! #BabcockUni',
        'Medical centre at UNIPORT only open 9-5? What about emergencies? This is ridiculous! #UNIPORTrants',
        'Classrooms at LASU are so cramped. No ventilation and projectors are ancient. Upgrade needed! #LASUstruggle',
        'AUL’s new ICT centre is a game-changer. Fast PCs and printing hubs. Love it! #AULtechie',
        'Why does UNIBEN Wi-Fi cut out every hour? Can’t even access ScienceDirect for research. #UNIBENwoes',
        'The social hub at Covenant is always buzzing. Perfect spot to chill between classes. #CovenantVibes',
        'UNILORIN cafeteria prices are insane! A plate of rice costs my whole allowance. #UNILORINrants',
        'New smart classrooms at UI are dope! Interactive boards and strong Wi-Fi. Learning made fun. #UIproud',
        'Hostel bathrooms at OAU are a health hazard. Leaking pipes and no hot water. #OAUfixit',
        'Labs at AUL have outdated equipment. How am I supposed to do practicals? #AULstruggles',
        'UNN’s new lecture theatre is massive and well-ventilated. Best place for classes! #UNNpraise',
        'Transportation at LASU is a joke. Waited 2 hours for a shuttle. #LASUprobs',
        'Babcock’s library Wi-Fi is super fast. Downloaded journals in seconds! #BabcockWins',
        'FUTO’s medical centre is basically non-existent. No staff after 6pm. #FUTOissues',
        'AUL’s new hostels are so modern! Clean, spacious, and Wi-Fi everywhere. #AULLife',
        'UNIBEN cafeteria food is overpriced and tasteless. Bring back affordability! #UNIBENrants',
        'Covenant’s ICT centre is a lifesaver for coding projects. 24/7 access! #CovenantTech',
        'UNILORIN classrooms are so hot. No fans or AC. Can’t focus! #UNILORINwoes',
        'UI’s new social hub is perfect for group study and chilling. #UIvibes',
        'UNN Wi-Fi is a myth. Haven’t connected once this semester. #UNNstruggles',
        'OAU’s library has no recent books. E-platforms are down half the time. #OAUfixit',
        'AUL’s medical centre saved me last night. 24/7 service is a blessing! #AULpraise',
        'LASU’s lecture halls are falling apart. Leaking roofs and broken chairs. #LASUprobs',
        'Babcock’s new hostel renovations are stunning. Feels like a hotel! #BabcockLife',
        'FUTO’s transportation system is chaotic. Buses always late. #FUTOwoes',
        'UNIBEN’s labs are top-tier. Modern equipment and open to students. #UNIBENwins',
        'Covenant’s cafeteria is affordable and has great options. Love it! #CovenantFood',
        'UNILORIN’s Wi-Fi is useless. Can’t even load Google. #UNILORINrants',
        'AUL’s smart classrooms make lectures so interactive. Best uni ever! #AULvibes',
        'UI’s restrooms are disgusting. No soap or water half the time. #UIissues',
        'UNN’s new ICT hub is amazing. Fast internet and cool vibe. #UNNpraise',
        'OAU’s hostels are overcrowded. No privacy at all. #OAUstruggles',
        'Babcock’s medical centre is always stocked and open. Saved me twice! #BabcockWins',
        'LASU’s library has no e-learning access. So outdated! #LASUwoes',
        'FUTO’s new lecture halls are spacious and cool. Love studying here! #FUTOpraise',
        'UNIBEN’s transportation is a mess. Missed a class waiting for a bus. #UNIBENprobs',
        'Covenant’s social hubs are perfect for networking. Always lively! #CovenantVibes',
        'UNILORIN’s labs are barely functional. Equipment from the 90s! #UNILORINrants',
        'AUL’s cafeteria is a bit pricey but the food is worth it. #AULLife',
        'UI’s Wi-Fi is so slow. Can’t even download lecture notes. #UIstruggles',
        'UNN’s new hostels are a dream. Clean, modern, and comfy! #UNNpraise',
        'OAU’s medical centre is understaffed. Waited hours for a check-up. #OAUissues'
    ],
    'username': [
        'student_unilag1', 'oaustudent22', 'aul_freshie', 'ui_girl01', 'covenant_geek', 'unn_student23',
        'abu_freshman', 'futo_engineer', 'babcock_bookworm', 'uniport_guy', 'lasu_lad', 'aul_coder',
        'uniben_scholar', 'covenant_chic', 'unilorin_girl', 'ui_student07', 'oaustudent09', 'aul_sciencenerd',
        'unn_guy01', 'lasu_engineer', 'babcock_reader', 'futo_medstudent', 'aul_freshie2', 'uniben_foodie',
        'covenant_dev', 'unilorin_scholar', 'ui_coolkid', 'unn_nerd', 'oau_booklover', 'aul_student03',
        'lasu_guy02', 'babcock_chic', 'futo_engineer2', 'uniben_sciencenerd', 'covenant_foodie', 'unilorin_coder',
        'aul_techie', 'ui_student08', 'unn_dev', 'oau_girl03', 'babcock_student', 'lasu_reader',
        'futo_scholar', 'uniben_engineer', 'covenant_networker', 'unilorin_sciencenerd', 'aul_foodie', 'ui_coder',
        'unn_freshie', 'oau_student04'
    ],
    'location': [
        'Lagos', 'Ifé', 'Urban', 'Ibadan', 'Ota', 'Nsukka', 'Zaria', 'Owerri', 'Ogun', 'Port Harcourt',
        'Lagos', 'Urban', 'Benin City', 'Ota', 'Ilorin', 'Ibadan', 'Ifé', 'Urban', 'Nsukka', 'Lagos',
        'Ogun', 'Owerri', 'Urban', 'Benin City', 'Ota', 'Ilorin', 'Ibadan', 'Nsukka', 'Ifé', 'Urban',
        'Lagos', 'Ogun', 'Owerri', 'Benin City', 'Ota', 'Ilorin', 'Urban', 'Ibadan', 'Nsukka', 'Ifé',
        'Ogun', 'Lagos', 'Owerri', 'Benin City', 'Ota', 'Ilorin', 'Urban', 'Ibadan', 'Nsukka', 'Ifé'
    ]
})

train_df = prepare_training_data()
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
y_train = train_df['sentiment']
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_split, y_train_split)
svm_val_predictions = svm_model.predict(X_val)
svm_metrics_report = classification_report(y_val, svm_val_predictions, output_dict=True)
svm_accuracy = accuracy_score(y_val, svm_val_predictions)
svm_model_metrics = {
    "accuracy": round(svm_accuracy, 4),
    "positive_precision": round(svm_metrics_report['positive']['precision'], 4),
    "positive_recall": round(svm_metrics_report['positive']['recall'], 4),
    "positive_f1_score": round(svm_metrics_report['positive']['f1-score'], 4),
    "negative_precision": round(svm_metrics_report['negative']['precision'], 4),
    "negative_recall": round(svm_metrics_report['negative']['recall'], 4),
    "negative_f1_score": round(svm_metrics_report['negative']['f1-score'], 4)
}

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)
rf_val_predictions = rf_model.predict(X_val)
rf_metrics_report = classification_report(y_val, rf_val_predictions, output_dict=True)
rf_accuracy = accuracy_score(y_val, rf_val_predictions)
rf_model_metrics = {
    "accuracy": round(rf_accuracy, 4),
    "positive_precision": round(rf_metrics_report['positive']['precision'], 4),
    "positive_recall": round(rf_metrics_report['positive']['recall'], 4),
    "positive_f1_score": round(rf_metrics_report['positive']['f1-score'], 4),
    "negative_precision": round(rf_metrics_report['negative']['precision'], 4),
    "negative_recall": round(rf_metrics_report['negative']['recall'], 4),
    "negative_f1_score": round(rf_metrics_report['negative']['f1-score'], 4)
}

def process_data(df, is_csv_input, excel_output_dir, chart_output_dir):
    infra_summary = pd.DataFrame() # Initialize as empty

    if is_csv_input:
        opinion_columns = [col for col in df.columns if col.startswith('My school')]
        infra_columns = [col for col in df.columns if 'How accessible is this infrastructure' in col]

        def create_text(row):
            return ' '.join([f"{col}: {row[col]}" for col in opinion_columns if pd.notna(row[col])])
        df['text'] = df.apply(create_text, axis=1)
        df['university_type'] = df['Kind of University'].apply(infer_university_type)
        df = df[df['university_type'] != 'unknown']

        if not df.empty and infra_columns:
            infra_summary = df.groupby('university_type')[infra_columns].mean().round(2)
    else:
        df['university_type'] = df['text'].apply(infer_university_type)
        df = df[df['university_type'] != 'unknown']

    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Only proceed with sentiment prediction if there's data left after filtering
    if not df.empty:
        X_test = vectorizer.transform(df['cleaned_text'])
        df['sentiment_svm'] = svm_model.predict(X_test)
        df['sentiment_rf'] = rf_model.predict(X_test)

        sentiment_summary = df.groupby(['university_type', 'sentiment_rf']).size().unstack(fill_value=0)
        sentiment_summary['total'] = sentiment_summary.sum(axis=1)
        for sentiment in ['positive', 'negative']:
            sentiment_summary[f'{sentiment}_percent'] = (sentiment_summary.get(sentiment, 0) / sentiment_summary['total'] * 100).round(2)
    else:
        sentiment_summary = pd.DataFrame() # Empty if no valid data

    chart_image_filename = 'sentiment_distribution.png' # Fixed name for easy access in Node.js
    chart_image_path = os.path.join(chart_output_dir, chart_image_filename)

    # Save sentiment distribution plot
    plt.figure(figsize=(10, 6))
    if not sentiment_summary.empty:
        melted_summary = pd.melt(
            sentiment_summary[['positive_percent', 'negative_percent']].reset_index(),
            id_vars=['university_type'],
            value_vars=['positive_percent', 'negative_percent'],
            var_name='variable',
            value_name='value'
        )
        sns.barplot(data=melted_summary, x='university_type', y='value', hue='variable', palette={'positive_percent': 'green', 'negative_percent': 'red'})
        plt.title('Sentiment Distribution by University Type (Random Forest)')
        plt.xlabel('University Type')
        plt.ylabel('Percentage')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(chart_image_path)
        plt.close()
    else:
        print("No valid data for plotting after filtering.", file=sys.stderr) # Send to stderr for Node.js logging

    # Convert summaries to dictionary for JSON output
    sentiment_summary_dict = sentiment_summary.to_dict(orient='index') if not sentiment_summary.empty else {}
    infra_summary_dict = infra_summary.to_dict(orient='index') if not infra_summary.empty else {}

    return df, sentiment_summary_dict, infra_summary_dict, chart_image_filename


if __name__ == '__main__':
    input_file_path = None
    is_csv_input_type = False
    excel_output_dir = None
    chart_output_dir = None

    # Parse command-line arguments
    # sys.argv[0] is script name
    # sys.argv[1] is input_file_path
    # sys.argv[2] is 'csv' or 'other'
    # sys.argv[3] is excel_output_dir
    # sys.argv[4] is chart_output_dir
    if len(sys.argv) > 4:
        input_file_path = sys.argv[1]
        is_csv_input_type = (sys.argv[2] == 'csv')
        excel_output_dir = sys.argv[3]
        chart_output_dir = sys.argv[4]
    else:
        # Fallback for direct testing or if not called from Node.js with all args
        print("Warning: Insufficient command-line arguments. Using sample data and default output paths.", file=sys.stderr)
        excel_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis_results', 'analysis_files')
        chart_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis_results', 'images')
        os.makedirs(excel_output_dir, exist_ok=True)
        os.makedirs(chart_output_dir, exist_ok=True)

    df_to_process = None
    if input_file_path and os.path.exists(input_file_path):
        try:
            df_to_process = pd.read_csv(input_file_path)
            if is_csv_input_type and not all(col in df_to_process.columns for col in ['Timestamp', 'Kind of University']):
                print(f"CSV file '{os.path.basename(input_file_path)}' missing 'Timestamp' or 'Kind of University' columns. Using sample X posts dataset.", file=sys.stderr)
                df_to_process = sample_data
                is_csv_input_type = False
        except Exception as e:
            print(f"Error loading uploaded CSV: {e}. Using sample X posts dataset.", file=sys.stderr)
            df_to_process = sample_data
            is_csv_input_type = False
    else:
        print("No valid input file provided or file not found. Using sample X posts dataset.", file=sys.stderr)
        df_to_process = sample_data
        is_csv_input_type = False

    df_processed, sentiment_summary_dict, infra_summary_dict, chart_image_filename = process_data(
        df_to_process.copy(),
        is_csv_input_type,
        excel_output_dir,
        chart_output_dir
    )

    output_excel_filename = f'analysis_result_{uuid.uuid4()}.xlsx' # Changed prefix for clarity
    output_file_path = os.path.join(excel_output_dir, output_excel_filename)

    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        if is_csv_input_type:
            cols_to_save = ['Timestamp', 'Username', 'Kind of University', 'text', 'university_type', 'sentiment_svm', 'sentiment_rf']
            # Filter columns that exist in the DataFrame
            df_processed[df_processed.columns.intersection(cols_to_save)].to_excel(writer, sheet_name='Raw_Data', index=False)
        else:
            df_processed[['created_at', 'username', 'text', 'university_type', 'sentiment_svm', 'sentiment_rf']].to_excel(writer, sheet_name='Raw_Data', index=False)

        if sentiment_summary_dict: # Only save if not empty
            sentiment_summary_df = pd.DataFrame.from_dict(sentiment_summary_dict, orient='index')
            sentiment_summary_df.index.name = 'university_type' # Set index name
            sentiment_summary_df.to_excel(writer, sheet_name='Sentiment_Summary')
        if infra_summary_dict: # Only save if not empty
            infra_summary_df = pd.DataFrame.from_dict(infra_summary_dict, orient='index')
            infra_summary_df.index.name = 'university_type' # Set index name
            infra_summary_df.to_excel(writer, sheet_name='Infrastructure_Summary')

    # Prepare final output for Node.js in JSON format
    final_output = {
        "output_excel_file": output_excel_filename,
        "chart_image_file": chart_image_filename,
        "sentiment_summary": sentiment_summary_dict,
        "infra_summary": infra_summary_dict,
        "model_metrics": {
            "svm": svm_model_metrics,
            "rf": rf_model_metrics
        }
    }
    # Print JSON to stdout for Node.js to capture
    print(json.dumps(final_output))

    sys.stdout.flush() # Ensure all output is sent before exiting
