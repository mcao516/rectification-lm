# coding=utf-8

from googleapiclient import discovery


API_KEY = ""  # put your key here
client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)
ATTRIBUTES = {
    'TOXICITY': {},
    'SEVERE_TOXICITY': {},
    'IDENTITY_ATTACK': {},
    'INSULT': {},
    'PROFANITY': {},
    'THREAT': {},
    'SEXUALLY_EXPLICIT': {},  # Experimental attributes
    'FLIRTATION': {},  # Experimental attributes
}


def get_perspective_scores(sentence):
    analyze_request = {
        'comment': {'text': sentence},
        'requestedAttributes': ATTRIBUTES
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return {k: v['summaryScore']['value'] for k, v in response['attributeScores'].items()}
