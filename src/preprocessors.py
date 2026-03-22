import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ContextDataFeatureEngineering(BaseEstimator, TransformerMixin):
    NUMERICAL_COLS = [
        'valence', 'instrumentalness', 'liveness', 'speechiness', 
        'danceability', 'acousticness', 'energy', 'mode', 'key'
    ]

    def __init__(self):
        self.language_map = LANGUAGE_MAPPING
        self.timezone_map = TIMEZONE_MAPPING
        self.column_map = CONTEXT_COLUMN_MAP
        self.means_ = {}

    def fit(self, X, y=None):
        for col in self.NUMERICAL_COLS:
            if col in X.columns:
                self.means_[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()

        if 'time_zone' in X.columns:
            X['time_zone'] = X['time_zone'].fillna('Undefined')

        for col, mean_val in self.means_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mean_val)

        X['lang'] = X['lang'].replace(self.language_map)
        X['tweet_lang'] = X['tweet_lang'].replace(self.language_map)
        X['time_zone'] = X['time_zone'].replace(self.timezone_map)

        if 'created_at' in X.columns:
            X['created_at'] = pd.to_datetime(X['created_at'])

        X = X.rename(columns=self.column_map)
        return X


LANGUAGE_MAPPING = {
    'en': 'English',
    'en-GB': 'English (UK)',
    'en-gb': 'English (UK)',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh-Hans': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'nl': 'Dutch',
    'tr': 'Turkish',
    'pl': 'Polish',
    'sv': 'Swedish',
    'fi': 'Finnish',
    'no': 'Norwegian',
    'da': 'Danish',
    'id': 'Indonesian',
    'in': 'India',
    'hu': 'Hungarian',
    'cs': 'Czech',
    'ca': 'Catalan',
    'eu': 'Basque',
    'gl': 'Galician',
    'uk': 'Ukrainian',
    'bg': 'Bulgarian',
    'bs': 'Bosnian',
    'cy': 'Welsh',
    'et': 'Estonian',
    'fa': 'Persian',
    'hr': 'Croatian',
    'ht': 'Haitian Creole',
    'is': 'Icelandic',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'ro': 'Romanian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'tl': 'Tagalog',
    'vi': 'Vietnamese',
    'und': 'Undefined',
    'xx-lc': 'Unknown/Hidden'
}

TIMEZONE_MAPPING = {
    # UTC -11:00 to -09:00
    "International Date Line West": "UTC-11:00",
    "Midway Island": "UTC-11:00",
    "Hawaii": "UTC-10:00",
    "Alaska": "UTC-09:00",

    # UTC -08:00 to -06:00 (North America West/Central)
    "Pacific Time (US & Canada)": "UTC-08:00",
    "Arizona": "UTC-07:00",
    "Chihuahua": "UTC-07:00",
    "Mountain Time (US & Canada)": "UTC-07:00",
    "Central America": "UTC-06:00",
    "Central Time (US & Canada)": "UTC-06:00",
    "Guadalajara": "UTC-06:00",
    "Mexico City": "UTC-06:00",
    "Monterrey": "UTC-06:00",
    "Saskatchewan": "UTC-06:00",

    # UTC -05:00 to -03:00 (Americas East)
    "America/Chicago": "UTC-05:00",
    "America/Detroit": "UTC-05:00",
    "America/New_York": "UTC-05:00",
    "Bogota": "UTC-05:00",
    "Eastern Time (US & Canada)": "UTC-05:00",
    "Indiana (East)": "UTC-05:00",
    "Lima": "UTC-05:00",
    "Quito": "UTC-05:00",
    "America/Sao_Paulo": "UTC-03:00",
    "Atlantic Time (Canada)": "UTC-04:00",
    "Caracas": "UTC-04:00",
    "La Paz": "UTC-04:00",
    "Santiago": "UTC-04:00",
    "Brasilia": "UTC-03:00",
    "Buenos Aires": "UTC-03:00",
    "Greenland": "UTC-03:00",
    "Newfoundland": "UTC-03:30",

    # UTC -02:00 to +00:00 (Atlantic / West Europe)
    "Mid-Atlantic": "UTC-02:00",
    "Azores": "UTC-01:00",
    "Casablanca": "UTC+00:00",
    "Dublin": "UTC+00:00",
    "Edinburgh": "UTC+00:00",
    "Europe/London": "UTC+00:00",
    "Lisbon": "UTC+00:00",
    "London": "UTC+00:00",

    # UTC +01:00 to +02:00 (Europe / Africa)
    "Amsterdam": "UTC+01:00",
    "Belgrade": "UTC+01:00",
    "Berlin": "UTC+01:00",
    "Bern": "UTC+01:00",
    "Bratislava": "UTC+01:00",
    "Brussels": "UTC+01:00",
    "Budapest": "UTC+01:00",
    "Copenhagen": "UTC+01:00",
    "Ljubljana": "UTC+01:00",
    "Madrid": "UTC+01:00",
    "Paris": "UTC+01:00",
    "Prague": "UTC+01:00",
    "Rome": "UTC+01:00",
    "Stockholm": "UTC+01:00",
    "Vienna": "UTC+01:00",
    "Warsaw": "UTC+01:00",
    "West Central Africa": "UTC+01:00",
    "Zagreb": "UTC+01:00",
    "Athens": "UTC+02:00",
    "Cairo": "UTC+02:00",
    "Helsinki": "UTC+02:00",
    "Istanbul": "UTC+02:00",
    "Jerusalem": "UTC+02:00",
    "Kyiv": "UTC+02:00",
    "Pretoria": "UTC+02:00",
    "Riga": "UTC+02:00",
    "Sofia": "UTC+02:00",
    "Vilnius": "UTC+02:00",

    # UTC +03:00 to +05:00 (Middle East / Russia / South Asia)
    "Baghdad": "UTC+03:00",
    "Europe/Minsk": "UTC+03:00",
    "Kuwait": "UTC+03:00",
    "Minsk": "UTC+03:00",
    "Moscow": "UTC+03:00",
    "Riyadh": "UTC+03:00",
    "St. Petersburg": "UTC+03:00",
    "Tehran": "UTC+03:30",
    "Abu Dhabi": "UTC+04:00",
    "Muscat": "UTC+04:00",
    "Tbilisi": "UTC+04:00",
    "Yerevan": "UTC+04:00",
    "Ekaterinburg": "UTC+05:00",
    "Islamabad": "UTC+05:00",
    "Mumbai": "UTC+05:30",
    "New Delhi": "UTC+05:30",
    "Chennai": "UTC+05:30",
    "Kathmandu": "UTC+05:45",

    # UTC +06:00 to +09:00 (Central / East Asia)
    "Almaty": "UTC+06:00",
    "Astana": "UTC+06:00",
    "Bangkok": "UTC+07:00",
    "Jakarta": "UTC+07:00",
    "Novosibirsk": "UTC+07:00",
    "Beijing": "UTC+08:00",
    "Hong Kong": "UTC+08:00",
    "Irkutsk": "UTC+08:00",
    "Kuala Lumpur": "UTC+08:00",
    "Singapore": "UTC+08:00",
    "Taipei": "UTC+08:00",
    "Perth": "UTC+08:00",
    "Osaka": "UTC+09:00",
    "Sapporo": "UTC+09:00",
    "Seoul": "UTC+09:00",
    "Tokyo": "UTC+09:00",
    "Yakutsk": "UTC+09:00",

    # UTC +10:00 to +12:00 (Oceania)
    "Brisbane": "UTC+10:00",
    "Canberra": "UTC+10:00",
    "Hobart": "UTC+10:00",
    "Melbourne": "UTC+10:00",
    "Pacific/Guam": "UTC+10:00",
    "Sydney": "UTC+10:00",
    "New Caledonia": "UTC+11:00",
    "Auckland": "UTC+12:00",
    "Wellington": "UTC+12:00",
    "Adelaide": "UTC+09:30",
    "Harare": "UTC+02:00",
    "Monrovia": "UTC+00:00",
    
    # Unmapped
    "Undefined": "Unknown"
}

CONTEXT_COLUMN_MAP = {
    "user_id": "user_id",
    "track_id": "track_id",
    "hashtag": "Hash tags",
    "created_at": "Timestamp of the activity",
    "score": "Calculated value for the post",
    "lang": "Primary language for the user profile",
    "tweet_lang": "Primary language for the specific post",
    "time_zone": "Regional setting for the user",
    "instrumentalness": "Probability that the track contains no vocals",
    "liveness": "Likelihood the recording was a live performance",
    "speechiness": "Presence of spoken words",
    "danceability": "Suitability for dancing based on tempo and beat",
    "valence": "Musical positiveness",
    "loudness": "Overall volume in decibels",
    "tempo": "Speed of the track in beats per minute",
    "acousticness": "Likelihood that the track is not electronic",
    "energy": "Perceived intensity and activity level",
    "mode": "Major or minor status of the track",
    "key": "Musical scale of the track",
    "rating": "User-assigned or system-generated preference score"
}
