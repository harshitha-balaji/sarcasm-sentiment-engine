from dataclasses import dataclass, field
import json
import pathlib
import re
from typing import List, Set, Dict, Any, Tuple
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class ModelDependencyError(Exception):
    """Custom exception raised when required external assets or NLP models are missing."""
    pass

@dataclass
class SentenceEvidence:
    """Stores structural evaluation and score for a single targeted sentence."""
    text: str
    sarcasm: str  # "Yes" or "No"
    label: str    # "POSITIVE", "NEGATIVE", or "NEUTRAL"
    score: float

@dataclass
class SentimentReport:
    """Stores the aggregated metrics and compilation of all extracted evidence blocks."""
    entity_name: str
    overall_stance: str
    overall_average: float
    evidence: List[SentenceEvidence] = field(default_factory=list)

class LexicalSentimentEngine:
    def __init__(self, 
                 settings_path: str = "engine_settings.json", 
                 lexicon_path: str = "domain_lexicons.json", 
                 default_domain: str = "political"):
                 
        # 1. Load Essential NLP Tools Safely
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise ModelDependencyError(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm"
            ) from e
            
        import nltk
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()

        self.tracking_pronouns: Set[str] = {
            "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"
        }

        # 2. Parse Engine Operational Settings File
        settings_file = pathlib.Path(settings_path)
        if not settings_file.exists():
            raise FileNotFoundError(f"Operational settings file not found at: {settings_path}")
            
        with open(settings_file, "r", encoding="utf-8") as f:
            settings_data = json.load(f)
            
        self.settings: Dict[str, Any] = settings_data["global_settings"]
        self.sarcasm_cues: Set[str] = set(settings_data["sarcasm_cues"])

        # 3. Parse and Hold Entire Knowledge Data Warehouse File (Lexicons)
        lexicon_file = pathlib.Path(lexicon_path)
        if not lexicon_file.exists():
            raise FileNotFoundError(f"Domain lexicon data file not found at: {lexicon_path}")
            
        with open(lexicon_file, "r", encoding="utf-8") as f:
            self.lexicon_warehouse = json.load(f)
            
        # Cache standard base VADER lexicon configuration for safety during dynamic swaps
        self.base_vader_lexicon = self.sia.lexicon.copy()
        
        # Trigger initialization configuration routing
        self._switch_domain(default_domain)

    def _switch_domain(self, domain_name: str) -> None:
        """Safely hot-swaps the underlying lexicon and target institutional synonyms."""
        if domain_name not in self.lexicon_warehouse:
            raise ValueError(f"Domain '{domain_name}' requested but not defined in lexicon warehouse data.")
            
        domain_assets = self.lexicon_warehouse[domain_name]
        self.institutional_synonyms: Set[str] = set(domain_assets["institutional_synonyms"])
        
        # Reset VADER back to baseline and layer the isolated domain specific metrics over it
        self.sia.lexicon = self.base_vader_lexicon.copy()
        self.sia.lexicon.update(domain_assets["lexicon"])

    def analyze_conditional_sarcasm(self, text: str, base_score: float) -> Tuple[bool, float]:
        """Detects and structurally flips conditional sarcasm using config-driven rules."""
        low_text = text.lower()
        
        # Dynamically build a regex pattern from the settings backdrop trigger words list
        triggers_pattern = "|".join(self.settings["negative_backdrop_triggers"])
        sarcasm_regex = rf"\bif\b.*\b({triggers_pattern})\b"
        
        sarcasm_condition = re.search(sarcasm_regex, low_text)
        
        if sarcasm_condition:
            multiplier: float = self.settings["conditional_multiplier"]
            return True, -abs(base_score if base_score != 0 else 0.5) * multiplier
            
        return False, base_score

    def get_evidence(self, text: str, entity_name: str) -> List[str]:
        """Extracts text evidence using contextual look-back and token-cleansed look-ahead configuration."""
        # Normalize non-standard unicode quotes to standard straight quotes before processing
        normalized_text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        
        doc = self.nlp(normalized_text)
        evidence: List[str] = []
        entity_found: bool = False
        
        # Dynamically fetch algorithmic settings from configuration dictionary
        memory_limit: int = self.settings["memory_limit"]
        counter: int = 0

        entity_exists_in_text: bool = entity_name.lower() in normalized_text.lower()

        # Pre-compile dynamic conditional pattern to optimize the sentence loop
        triggers_pattern = "|".join(self.settings["negative_backdrop_triggers"])
        conditional_sarcasm_regex = rf"\bif\b.*\b({triggers_pattern})\b"

        for sent in doc.sents:
            sent_text: str = sent.text.strip()
            if not sent_text:
                continue
            low_sent: str = sent_text.lower()
            
            # Condition A: Directly mentions target entity
            if entity_name.lower() in low_sent:
                evidence.append(sent_text)
                entity_found = True
                counter = memory_limit  
                
            # Condition B: Context setup (Institutional synonym OR global conditional sarcasm statement)
            elif entity_exists_in_text and not entity_found:
                has_synonym = any(syn in low_sent for syn in self.institutional_synonyms)
                has_global_conditional = bool(re.search(conditional_sarcasm_regex, low_sent))
                
                if has_synonym or has_global_conditional:
                    evidence.append(sent_text)
                
            # Condition C: Look-ahead dynamic tracking window
            elif entity_found and counter > 0:
                # Stripping leading quote symbols/spaces completely out of the regex matching target
                cleansed_start = re.sub(r"^['\"]+", "", low_sent).strip()
                
                # --- DYNAMIC PRONOUN ISOLATION FILTER ---
                # Categorize the target type dynamically using warehouse assets to prevent look-ahead pollution
                institutional_keywords = list(self.institutional_synonyms)
                is_institutional_target = any(k in entity_name.lower() for k in institutional_keywords)
                
                # Filter down active pronouns dynamically based on target structural type
                if is_institutional_target:
                    # Institutions are referred to as 'it' or collectively as 'they'
                    allowed_pronouns = {p for p in self.tracking_pronouns if p in ["it", "its", "they", "them", "their", "theirs"]}
                else:
                    # Individuals use all pronouns, but prioritize personal tags
                    allowed_pronouns = self.tracking_pronouns
                
                # Match pronouns at the absolute beginning of the word string boundary
                start_pronoun_pattern = rf"^(" + "|".join(allowed_pronouns) + r")\b"
                has_start_pronoun = bool(re.search(start_pronoun_pattern, cleansed_start))
                
                # Match pronoun attribution trailing a closed quote layer
                post_quote_pattern = rf"['\",“’‘”\s]+(" + "|".join(allowed_pronouns) + r")\b"
                has_trailing_pronoun_attribution = bool(re.search(post_quote_pattern, low_sent))
                
                has_valid_pronoun_subject = has_start_pronoun or has_trailing_pronoun_attribution
                # ----------------------------------------
                
                # Check for structural discourse pivots
                transition_markers = ["however", "nevertheless", "nonetheless", "furthermore", "moreover", "but", "meanwhile"]
                transition_pattern = rf"^(" + "|".join(transition_markers) + r")\b"
                has_discourse_transition = bool(re.search(transition_pattern, cleansed_start))
                
                has_sarcasm_cue = any(cue in low_sent for cue in self.sarcasm_cues)
                has_structural_sarcasm = "!!" in low_sent and any(word.isupper() and len(word) > 1 for word in sent_text.split())
                has_explicit_entity_mention = entity_name.lower() in low_sent
                
                if has_valid_pronoun_subject or has_discourse_transition or has_sarcasm_cue or has_structural_sarcasm or has_explicit_entity_mention:
                    evidence.append(sent_text)
                    counter = memory_limit  # Reset tracking window memory
                else:
                    counter -= 1  # Countdown stepping through the lookahead window
                    
        return evidence

    def process_article(self, text: str, entity_name: str, domain: str = "political") -> SentimentReport:
        """Orchestrates pipeline execution and returns an immutable data contract container."""
        # Dynamically route custom lexicon dependencies before initializing the pipeline passes
        self._switch_domain(domain)
        
        evidence_sentences = self.get_evidence(text, entity_name)
        
        # Instantiate report container data architecture
        report = SentimentReport(
            entity_name=entity_name,
            overall_stance="NEUTRAL",
            overall_average=0.0
        )
        
        if not evidence_sentences:
            return report

        total_score = 0.0
        sentiment_threshold: float = self.settings["sentiment_threshold"]
        irony_floor_penalty: float = self.settings["irony_floor_penalty"]
        conditional_multiplier: float = self.settings["conditional_multiplier"]
        
        # Track active irony downstream across the paragraph state machine
        active_irony_context = False

        for sent in evidence_sentences:
            low_sent = sent.lower()
            
            # --- DYNAMIC CLAUSE ISOLATION PASS ---
            # If a sentence contains a cross-entity split, isolate the clause containing the target
            scoring_target = sent
            if "," in sent or ";" in sent:
                # Split along common structural conjunction points
                clauses = re.split(r'[,;]\s+(?:while|but|whereas|although|the|as|their|they|she|he)\b|[,;]', sent, flags=re.IGNORECASE)
                for clause in clauses:
                    if clause and entity_name.lower() in clause.lower():
                        scoring_target = clause.strip()
                        break
            
            # Run standard lexical scoring over the isolated target clause
            base_score: float = self.sia.polarity_scores(scoring_target)['compound']
            # -------------------------------------
            
            # Layered context modifications
            sarcasm_detected, modified_score = self.analyze_conditional_sarcasm(sent, base_score)
            
            # Context evaluation fallback checks
            if not sarcasm_detected:
                has_explicit_cue = any(cue in low_sent for cue in self.sarcasm_cues)
                has_structural_sarcasm = "!!" in low_sent and any(w.isupper() and len(w) > 1 for w in sent.split())
                
                # Native domain quote modifiers
                has_mocking_attribution = any(anchor in low_sent for anchor in ["straight face", "mocked", "hollow"])
                
                if (has_explicit_cue and base_score > 0) or has_structural_sarcasm or (has_mocking_attribution and base_score > 0):
                    sarcasm_detected = True
                    modified_score = irony_floor_penalty if base_score <= 0.5 else -base_score
                
                # Contextual Interception Layer: Check if we are inside an active ironic stream
                elif active_irony_context and base_score > 0:
                    sarcasm_detected = True
                    modified_score = -abs(base_score) * conditional_multiplier

            # Update the downstream irony context state flag
            if sarcasm_detected:
                active_irony_context = True

            # Classify current sentence metric label
            if modified_score > sentiment_threshold:
                label = "POSITIVE"
            elif modified_score < -sentiment_threshold:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
                
            total_score += modified_score
            
            # Append structured sentence evidence to the report
            report.evidence.append(
                SentenceEvidence(
                    text=sent,
                    sarcasm="Yes" if sarcasm_detected else "No",
                    label=label,
                    score=round(modified_score, 4)
                )
            )

        # Final metrics aggregation assembly
        report.overall_average = round(total_score / len(evidence_sentences), 4)
        
        if report.overall_average > sentiment_threshold:
            report.overall_stance = "POSITIVE"
        elif report.overall_average < -sentiment_threshold:
            report.overall_stance = "NEGATIVE"
            
        return report


if __name__ == "__main__":
    # --- INTERACTIVE USER INPUT TEST ---
    print("=" * 60)
    print("        LEXICAL SENTIMENT ENGINE TERMINAL TESTER        ")
    print("=" * 60)
    
    print("Initializing engine and parsing configuration files...")
    try:
        engine = LexicalSentimentEngine()
        print("Engine initialized successfully!\n")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        exit(1)

    # Gather text input from user
    print("Enter or paste your article text below:")
    user_article = input("> ").strip()
    
    if not user_article:
        print("Error: Article text cannot be blank.")
        exit(1)

    # Gather target entity name from user
    print("\nWhat is the specific target entity name you want to track?")
    user_target = input("> ").strip()
    
    if not user_target:
        print("Error: Target entity name cannot be blank.")
        exit(1)

    # Gather target operational domain configuration routing from user
    available_domains = list(engine.lexicon_warehouse.keys())
    print(f"\nSpecify operational evaluation domain {available_domains}:")
    user_domain = input("> ").strip().lower()
    
    if user_domain not in engine.lexicon_warehouse.keys():
        print("Invalid choice. Defaulting processing sequence to 'political' context boundaries.")
        user_domain = "political"

    # Run execution pipeline
    print(f"\nProcessing evaluations for target entity: '{user_target}' using [{user_domain}] rules...")
    report: SentimentReport = engine.process_article(user_article, user_target, domain=user_domain)

    # --- PRESENTATION LAYER ---
    print("\n" + "=" * 80)
    print(f"SENTIMENT ANALYSIS REPORT FOR: {report.entity_name.upper()}")
    print(f"OVERALL STANCE : {report.overall_stance}")
    print(f"SCORE AVERAGE  : {report.overall_average}")
    print("=" * 80)
    
    if not report.evidence:
        print("\n[!] No matching context blocks or entity tracking tokens found in this text.")
    else:
        print("\nEXTRACTED EVIDENCE DETAILS:")
        for idx, ev in enumerate(report.evidence, 1):
            print(f"\n[{idx}] Sentence: \"{ev.text}\"")
            print(f"    Sarcasm Detected: {ev.sarcasm}")
            print(f"    Sentence Label  : {ev.label}")
            print(f"    Calculated Score: {ev.score}")
            
    print("\n" + "=" * 80)