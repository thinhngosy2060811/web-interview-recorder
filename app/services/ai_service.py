import whisper
import google.generativeai as genai
import asyncio
import logging
import json
import statistics
import subprocess
from typing import Dict, List, Optional
from pathlib import Path

# Import các biến cấu hình từ file config
from app.config import GEMINI_API_KEY, LAYER1_THRESHOLDS, WPM_RANGES, FILLER_WORDS
# Import hàm tiện ích
from app.utils import get_bangkok_timestamp
# Import hàm update metadata từ file service bên cạnh
from app.services.file_service import update_metadata

logger = logging.getLogger(__name__)

WHISPER_MODEL = None
gemini_model = None

def init_ai_models():
    global WHISPER_MODEL, gemini_model
    
    # 1. Load Whisper
    try:
        logger.info("Loading Whisper model (small)...")
        WHISPER_MODEL = whisper.load_model("small")
        logger.info("✅ Whisper loaded")
    except Exception as e:
        logger.error(f"❌ Whisper failed: {e}")

    # 2. Load Gemini
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Cấu hình safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            gemini_model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)
            logger.info("✅ Gemini initialized")
        except Exception as e:
            logger.error(f"❌ Gemini failed: {e}")
    

def calculate_filler_density(transcript_text: str) -> float:
    """
    Tính mật độ từ đệm (Filler Density)
    Returns: Percentage (0-100)
    """
    text_lower = transcript_text.lower()
    words = text_lower.split()
    
    if len(words) == 0:
        return 0.0
    
    filler_count = sum(1 for word in words if word in FILLER_WORDS)
    
    # Check multi-word fillers
    for filler in ['you know', 'i mean']:
        filler_count += text_lower.count(filler)
    
    density = (filler_count / len(words)) * 100
    return round(density, 2)


def calculate_silence_ratio(segments: List[Dict]) -> tuple:
    """
    Tính tỷ lệ im lặng (Silence Ratio)
    Returns: (silence_ratio_percentage, total_pause_time, num_pauses)
    """
    if not segments or len(segments) < 2:
        return 0.0, 0.0, 0
    
    total_pause_time = 0.0
    num_pauses = 0
    
    for i in range(1, len(segments)):
        gap = segments[i]['start'] - segments[i-1]['end']
        if gap > 1.5:  # Ngưỡng 1.5 giây
            total_pause_time += gap
            num_pauses += 1
    
    total_duration = segments[-1]['end'] - segments[0]['start']
    
    if total_duration == 0:
        return 0.0, 0.0, 0
    
    silence_ratio = (total_pause_time / total_duration) * 100
    return round(silence_ratio, 2), round(total_pause_time, 2), num_pauses


def calculate_speaking_rate(word_count: int, duration_seconds: float) -> tuple:
    """
    Tính tốc độ nói (Speaking Rate)
    Returns: (wpm, category)
    """
    if duration_seconds == 0:
        return 0, "unknown"
    
    duration_minutes = duration_seconds / 60
    wpm = word_count / duration_minutes
    
    if wpm < WPM_RANGES["slow"][1]:
        category = "slow"
    elif wpm < WPM_RANGES["good"][1]:
        category = "good"
    else:
        category = "fast"
    
    return round(wpm, 1), category


def analyze_layer1_metrics(transcript_data: Dict, focus_score: int) -> Dict:
    """
    Phân tích Layer 1 - Quantitative Metrics
    
    Args:
        transcript_data: Dict chứa 'text' và 'segments' từ Whisper
        focus_score: Focus score từ Frontend (0-100)
    
    Returns:
        Dict chứa tất cả metrics và decision
    """
    text = transcript_data.get('text', '')
    segments = transcript_data.get('segments', [])
    
    # 1. Word count
    words = text.split()
    word_count = len(words)
    
    # 2. Filler density
    filler_density = calculate_filler_density(text)
    
    # 3. Silence ratio
    silence_ratio, pause_time, num_pauses = calculate_silence_ratio(segments)
    
    # 4. Speaking rate
    duration = segments[-1]['end'] if segments else 0
    wpm, wpm_category = calculate_speaking_rate(word_count, duration)
    
    # 5. Decision
    is_bad = (
        word_count < LAYER1_THRESHOLDS["min_word_count"] or
        silence_ratio > LAYER1_THRESHOLDS["max_silence_ratio"] * 100 or
        focus_score < LAYER1_THRESHOLDS["min_focus_score"]
    )
    
    result = {
        "word_count": word_count,
        "filler_density_percent": filler_density,
        "silence_ratio_percent": silence_ratio,
        "total_pause_seconds": pause_time,
        "num_pauses": num_pauses,
        "speaking_rate_wpm": wpm,
        "wpm_category": wpm_category,
        "focus_score": focus_score,
        "duration_seconds": round(duration, 2),
        "flagged_as_bad": is_bad,
        "flag_reasons": []
    }
    
    # Ghi lý do flag
    if word_count < LAYER1_THRESHOLDS["min_word_count"]:
        result["flag_reasons"].append(f"Too short ({word_count} words < {LAYER1_THRESHOLDS['min_word_count']})")
    
    if silence_ratio > LAYER1_THRESHOLDS["max_silence_ratio"] * 100:
        result["flag_reasons"].append(f"Too much silence ({silence_ratio}% > {LAYER1_THRESHOLDS['max_silence_ratio']*100}%)")
    
    if focus_score < LAYER1_THRESHOLDS["min_focus_score"]:
        result["flag_reasons"].append(f"Low focus ({focus_score}% < {LAYER1_THRESHOLDS['min_focus_score']}%)")
    
    return result

async def analyze_layer2_ai(
    question_text: str,
    transcript_text: str,
    layer1_metrics: Dict
) -> Dict:
    """
    Phân tích Layer 2 - AI Semantic Analysis using Gemini 2.5 Flash
    
    Args:
        question_text: Câu hỏi được hỏi
        transcript_text: Transcript đầy đủ
        layer1_metrics: Kết quả từ Layer 1
    
    Returns:
        Dict chứa priority, reason, và AI analysis
    """
    
    # ============== VALIDATE KEY VÀ MODEL ==============
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not available")
        return {
            "priority": "UNKNOWN",
            "reason": "There no have API key",
            "ai_available": False,
            "error": "GEMINI_API_KEY not set"
        }
    
    if not gemini_model:
        logger.error("❌ Gemini model not initialized")
        return {
            "priority": "UNKNOWN",
            "reason": "AI model is not generate",
            "ai_available": False,
            "error": "Gemini model is None"
        }
    
    try:
        # ✅ THÊM SAFETY SETTINGS
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            safety_settings=safety_settings
        )
        
        logger.info("🤖 Calling Gemini for Layer 2 analysis...")
        
        # ============== CONSTRUCT PROMPT ==============
        prompt = f"""You are an expert HR interviewer evaluating a Data Analyst candidate's video interview response.

**QUESTION ASKED:**
{question_text}

**CANDIDATE'S TRANSCRIPT:**
{transcript_text}

**QUANTITATIVE METRICS (Layer 1):**
- Word Count: {layer1_metrics['word_count']}
- Filler Density: {layer1_metrics['filler_density_percent']}%
- Silence Ratio: {layer1_metrics['silence_ratio_percent']}%
- Speaking Rate: {layer1_metrics['speaking_rate_wpm']} WPM ({layer1_metrics['wpm_category']})
- Focus Score: {layer1_metrics['focus_score']}%
- Flagged as Bad: {layer1_metrics['flagged_as_bad']}

**YOUR TASK:**
Evaluate the candidate's response and assign a priority level for HR review.
You must answer briefly but fully and to the point

**EVALUATION CRITERIA:**
1. **Content Relevance**: Does the answer address the question directly?
2. **Communication Skills**: Clear structure, fluency, confidence
3. **Professionalism**: Appropriate tone and demeanor
4. **Technical Depth** (if applicable): Shows understanding of concepts

**OUTPUT FORMAT (JSON ONLY):**
{{
  "priority": "HIGH" | "MEDIUM" | "LOW",
  "reason": "Brief 1-2 sentence explanation in English for HR",
  "content_score": 0-10,
  "communication_score": 0-10,
  "overall_impression": "positive" | "neutral" | "negative"
}}

**PRIORITY GUIDELINES:**
- HIGH: Strong candidate, clear answers, good communication, relevant experience
- MEDIUM: Acceptable but has some weaknesses, needs closer review
- LOW: Poor answer quality, irrelevant content, or major communication issues

IMPORTANT: Respond with ONLY the JSON object. No markdown, no extra text."""

        # ============== CALL GEMINI API ==============
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        # ============== CHECK FINISH REASON ==============
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = response.candidates[0].finish_reason
            
            if finish_reason == 3:  # SAFETY blocked
                logger.warning("⚠️ Response blocked by safety filter")
                return {
                    "priority": "NOT EVALUATED",
                    "reason": "Cannot be analyzed due to security filter.",
                    "ai_available": False,
                    "error": "SAFETY_BLOCKED"
                }
            elif finish_reason != 1:  # Not STOP
                logger.warning(f"⚠️ Unexpected finish_reason: {finish_reason}")
        
        # ============== PARSE RESPONSE ==============
        response_text = response.text.strip()
        logger.info(f"📥 Raw Gemini response (first 200 chars): {response_text[:200]}")
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        ai_result = json.loads(response_text)
        
        # ============== VALIDATE REQUIRED FIELDS ==============
        required_fields = ["priority", "reason"]
        for field in required_fields:
            if field not in ai_result:
                raise ValueError(f"Missing required field: {field}")
        
        # Set defaults for optional fields
        ai_result.setdefault("content_score", 5)
        ai_result.setdefault("communication_score", 5)
        ai_result.setdefault("overall_impression", "neutral")
        
        logger.info(f"✅ AI Analysis completed - Priority: {ai_result.get('priority', 'UNKNOWN')}")
        
        return {
            **ai_result,
            "ai_available": True,
            "model_used": "gemini-2.5-flash",
            "tokens_used": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse Gemini response: {e}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
        return {
            "priority": "NOT EVALUATED",
            "reason": "AI parsing error (JSON parse error), requires manual review.",
            "ai_available": False,
            "error": f"JSON parse error: {str(e)}",
            "raw_response": response_text[:500] if 'response_text' in locals() else None
        }
    
    except Exception as e:
        logger.error(f"❌ Gemini API error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return {
            "priority": "NOT EVALUATED",
            "reason": "AI system error, manual review required.",
            "ai_available": False,
            "error": str(e)
        }
        
async def calculate_final_ranking(folder_path: Path):
    """
    Tính toán ranking tổng hợp sau khi hoàn thành tất cả câu hỏi
    """
    meta_file = folder_path / "meta.json"
    
    if not meta_file.exists():
        logger.error("❌ Metadata file not found for final ranking")
        return
    
    try:
        with meta_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        questions = metadata.get("questions", [])
        
        if not questions:
            logger.warning("⚠️ No questions data for ranking")
            return
        
        # ============== COLLECT METRICS ==============
        all_metrics = {
            "word_counts": [],
            "filler_densities": [],
            "silence_ratios": [],
            "wpms": [],
            "focus_scores": [],
            "priorities": [],
            "content_scores": [],
            "communication_scores": []
        }
        
        valid_questions = 0
        
        for q in questions:
            metrics = q.get("metrics", {})
            ai_eval = q.get("ai_evaluation", {})
            
            if metrics:
                all_metrics["word_counts"].append(metrics.get("word_count", 0))
                all_metrics["filler_densities"].append(metrics.get("filler_density_percent", 0))
                all_metrics["silence_ratios"].append(metrics.get("silence_ratio_percent", 0))
                all_metrics["wpms"].append(metrics.get("speaking_rate_wpm", 0))
                all_metrics["focus_scores"].append(metrics.get("focus_score", 0))
                valid_questions += 1
            
            if ai_eval and ai_eval.get("ai_available"):
                priority = ai_eval.get("priority", "MEDIUM")
                all_metrics["priorities"].append(priority)
                all_metrics["content_scores"].append(ai_eval.get("content_score", 5))
                all_metrics["communication_scores"].append(ai_eval.get("communication_score", 5))
        
        # ============== CALCULATE AVERAGES ==============
        def safe_mean(lst):
            return round(statistics.mean(lst), 2) if lst else 0
        
        summary = {
            "total_questions_analyzed": valid_questions,
            "avg_word_count": safe_mean(all_metrics["word_counts"]),
            "avg_filler_density_percent": safe_mean(all_metrics["filler_densities"]),
            "avg_silence_ratio_percent": safe_mean(all_metrics["silence_ratios"]),
            "avg_wpm": safe_mean(all_metrics["wpms"]),
            "avg_focus_score": safe_mean(all_metrics["focus_scores"]),
            "avg_content_score": safe_mean(all_metrics["content_scores"]),
            "avg_communication_score": safe_mean(all_metrics["communication_scores"])
        }
        
        # ============== DETERMINE FINAL PRIORITY ==============
        priorities = all_metrics["priorities"]
        
        if not priorities:
            logger.warning("⚠️ No AI priorities available for final ranking")
            summary["final_priority"] = "UNKNOWN"
            summary["final_reason"] = "There is no AI data to evaluate."
        else:
            high_count = priorities.count("HIGH")
            low_count = priorities.count("LOW")
            
            if high_count >= len(priorities) * 0.6:
                final_priority = "HIGH"
                final_reason = "Outstanding candidate - Many high-quality answers"
            elif low_count >= len(priorities) * 0.5:
                final_priority = "LOW"
                final_reason = "Weak candidates - Many poor-quality answers"
            else:
                final_priority = "MEDIUM"
                final_reason = "Average candidate - Needs closer consideration"
            
            summary["final_priority"] = final_priority
            summary["final_reason"] = final_reason
            summary["priority_distribution"] = {
                "HIGH": high_count,
                "MEDIUM": priorities.count("MEDIUM"),
                "LOW": low_count
            }
        
        # ============== GEMINI OVERALL SUMMARY ==============
        if GEMINI_API_KEY and gemini_model:
            try:
                # ✅ THÊM SAFETY SETTINGS
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
                
                model = genai.GenerativeModel(
                    'gemini-2.5-flash',
                    safety_settings=safety_settings
                )
                
                logger.info("🤖 Generating overall AI summary with Gemini...")
                
                # Ghép tất cả transcript lại
                full_transcript = ""
                for i, q in enumerate(questions, 1):
                    transcript_file = folder_path / q.get("transcriptFile", "")
                    if transcript_file.exists():
                        content = transcript_file.read_text(encoding='utf-8')
                        text_only = content.split("--- TIMESTAMPS ---")[0].strip()
                        full_transcript += f"\n\n--- QUESTION {i} ---\n{text_only}"
                
                # Truncate nếu quá dài (Gemini có limit context)
                max_transcript_length = 3000
                if len(full_transcript) > max_transcript_length:
                    full_transcript = full_transcript[:max_transcript_length] + "\n\n[...truncated...]"
                
                # Prompt cho overall summary
                overall_prompt = f"""You are reviewing a complete Data Analyst interview. Provide a concise overall assessment.

**INTERVIEW STATISTICS:**
{json.dumps(summary, indent=2, ensure_ascii=False)}

**FULL TRANSCRIPT (SAMPLE):**
{full_transcript}

**YOUR TASK:**
Provide a brief overall assessment (2-3 sentences in English) for HR to decide if they should watch the videos.

**OUTPUT FORMAT (JSON ONLY):**
{{
  "overall_summary": "2-3 sentence summary in English",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "recommendation": "RECOMMEND" | "NEUTRAL" | "NOT_RECOMMEND"
}}

Respond with ONLY the JSON object. No markdown, no extra text."""

                # ✅ GỌI GEMINI
                response = await asyncio.to_thread(
                    model.generate_content,
                    overall_prompt,
                    generation_config={
                        "temperature": 0.3, 
                        "max_output_tokens": 2048,
                        "top_p": 0.95
                    }
                )
                
                # ✅ KIỂM TRA FINISH REASON
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    
                    if finish_reason == 3:  # SAFETY blocked
                        logger.warning("⚠️ Overall summary blocked by safety filter")
                        summary["overall_ai_summary"] = {
                            "error": "SAFETY_BLOCKED",
                            "message": "Nội dung bị chặn bởi bộ lọc an toàn"
                        }
                        # Continue to save other data
                    elif finish_reason == 1:  # SUCCESS
                        response_text = response.text.strip()
                        logger.info(f"📥 Gemini overall response (first 200 chars): {response_text[:200]}")
                        
                        # Clean markdown code blocks
                        if "```json" in response_text:
                            response_text = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            response_text = response_text.split("```")[1].split("```")[0].strip()
                        
                        # Parse JSON
                        overall_analysis = json.loads(response_text)
                        
                        # Validate
                        required_fields = ["overall_summary", "strengths", "weaknesses", "recommendation"]
                        for field in required_fields:
                            if field not in overall_analysis:
                                logger.warning(f"⚠️ Missing field in AI response: {field}")
                                overall_analysis[field] = "N/A" if field in ["overall_summary", "recommendation"] else []
                        
                        summary["overall_ai_summary"] = overall_analysis
                        logger.info(f"✅ Overall AI summary generated - Recommendation: {overall_analysis.get('recommendation', 'N/A')}")
                    else:
                        logger.warning(f"⚠️ Unexpected finish_reason: {finish_reason}")
                        summary["overall_ai_summary"] = {
                            "error": f"Unexpected finish_reason: {finish_reason}"
                        }
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parse error in overall summary: {e}")
                logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
                summary["overall_ai_summary"] = {
                    "error": "JSON parse error",
                    "raw_response": response_text[:500] if 'response_text' in locals() else None
                }
            
            except Exception as e:
                logger.error(f"❌ Error generating overall summary: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                summary["overall_ai_summary"] = {
                    "error": str(e)
                }
        else:
            logger.warning("⚠️ Gemini not available for final ranking")
            logger.warning(f"API Key exists: {bool(GEMINI_API_KEY)}, Model exists: {bool(gemini_model)}")
            summary["overall_ai_summary"] = {
                "error": "Gemini API not configured",
                "details": f"API_KEY={'SET' if GEMINI_API_KEY else 'MISSING'}, MODEL={'INITIALIZED' if gemini_model else 'NOT_INITIALIZED'}"
            }
        
        # ============== MERGE ALL TRANSCRIPTS ==============
        full_transcript_content = "=== FULL INTERVIEW TRANSCRIPT ===\n\n"
        full_transcript_content += f"Candidate: {metadata.get('userName')}\n"
        full_transcript_content += f"Session: {metadata.get('sessionStartTime')}\n\n"
        
        for i, q in enumerate(questions, 1):
            transcript_file = folder_path / q.get("transcriptFile", "")
            if transcript_file.exists():
                full_transcript_content += f"\n{'='*60}\n"
                full_transcript_content += transcript_file.read_text(encoding='utf-8') + "\n"
        
        full_transcript_file = folder_path / "FULL_TRANSCRIPT.txt"
        full_transcript_file.write_text(full_transcript_content, encoding='utf-8')
        logger.info(f"📄 Full transcript saved: {full_transcript_file.name}")
        
        # ============== UPDATE METADATA ==============
        metadata["final_ranking_summary"] = summary
        metadata["final_ranking_calculated_at"] = get_bangkok_timestamp()
        
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Final ranking calculated - Priority: {summary.get('final_priority', 'UNKNOWN')}")
        logger.info(f"📊 Summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"❌ Error calculating final ranking: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
async def transcribe_video_whisper(video_path: Path, question_index: int) -> Optional[str]:
    """Chuyển đổi Video sang Text dùng Whisper"""
    if WHISPER_MODEL is None:
        return None
    
    try:
        # 1. Extract audio
        audio_path = video_path.with_suffix('.wav')
        subprocess.run([
            'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', str(audio_path), '-y', '-loglevel', 'error'
        ], check=False, timeout=60)
        
        # 2. Transcribe (Chạy trong thread riêng để không block server)
        logger.info(f"🎤 Transcribing Q{question_index}...")
        whisper_result = await asyncio.to_thread(
            WHISPER_MODEL.transcribe,
            str(audio_path),
            language='en',
            fp16=False
        )
        
        # 3. Format Transcript
        transcript_text = f"=== TRANSCRIPT Q{question_index} ===\n"
        transcript_text += f"Time: {get_bangkok_timestamp()}\n\n"
        transcript_text += whisper_result['text'].strip() + "\n\n"
        transcript_text += "--- TIMESTAMPS ---\n"
        
        for segment in whisper_result.get('segments', []):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            transcript_text += f"[{start//60:02.0f}:{start%60:05.2f} -> {end//60:02.0f}:{end%60:05.2f}] {text}\n"
            
        # 4. Save to file
        transcript_file = video_path.parent / f"Q{question_index}_transcript.txt"
        transcript_file.write_text(transcript_text, encoding='utf-8')
        logger.info(f"📝 Transcript saved: {transcript_file.name}")
        
        # Cleanup
        audio_path.unlink(missing_ok=True)
        return whisper_result['text'].strip()

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return None
 
async def background_transcribe(folder_path: Path, video_path: Path, question_index: int, question_text: str, focus_score: int):
    """
    Hàm chạy background để:
    1. Transcribe video
    2. Phân tích Layer 1 (Quantitative)
    3. Phân tích Layer 2 (AI Semantic)
    4. Cập nhật metadata
    """
    try:
        logger.info(f"🎤 [Background] Starting analysis for Q{question_index}...")
        
        if WHISPER_MODEL is None:
            logger.warning(f"⚠️ Whisper model not loaded, skipping Q{question_index}")
            return
        
        # ============== STEP 1: TRANSCRIBE ==============
        audio_path = video_path.with_suffix('.wav')
        subprocess.run([
            'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', str(audio_path), '-y', '-loglevel', 'error'
        ], check=False, timeout=60)
        
        logger.info(f"🎤 Transcribing Q{question_index}...")
        whisper_result = await asyncio.to_thread(
            WHISPER_MODEL.transcribe,
            str(audio_path),
            language='en',
            fp16=False
        )
        
        transcript_text = whisper_result['text'].strip()
        segments = whisper_result.get('segments', [])
        
        # Save transcript
        transcript_content = f"=== TRANSCRIPT Q{question_index} ===\n"
        transcript_content += f"Question: {question_text}\n"
        transcript_content += f"Time: {get_bangkok_timestamp()}\n\n"
        transcript_content += transcript_text + "\n\n"
        transcript_content += "--- TIMESTAMPS ---\n"
        
        for segment in segments:
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            transcript_content += f"[{start//60:02.0f}:{start%60:05.2f} -> {end//60:02.0f}:{end%60:05.2f}] {text}\n"
        
        transcript_file = video_path.parent / f"Q{question_index}_transcript.txt"
        transcript_file.write_text(transcript_content, encoding='utf-8')
        logger.info(f"📄 Transcript saved: {transcript_file.name}")
        
        # ============== STEP 2: LAYER 1 ANALYSIS ==============
        logger.info(f"📊 Running Layer 1 analysis for Q{question_index}...")
        
        layer1_metrics = analyze_layer1_metrics(
            transcript_data={'text': transcript_text, 'segments': segments},
            focus_score=focus_score
        )
        
        logger.info(f"✅ Layer 1 completed - Flagged: {layer1_metrics['flagged_as_bad']}")
        
        # ============== STEP 3: LAYER 2 AI ANALYSIS ==============
        logger.info(f"🤖 Running Layer 2 AI analysis for Q{question_index}...")
        
        layer2_result = await analyze_layer2_ai(
            question_text=question_text,
            transcript_text=transcript_text,
            layer1_metrics=layer1_metrics
        )
        
        logger.info(f"✅ Layer 2 completed - Priority: {layer2_result.get('priority', 'UNKNOWN')}")
        
        # ============== STEP 4: UPDATE METADATA ==============
        analysis_summary = {
            "transcriptionStatus": "completed",
            "transcriptFile": f"Q{question_index}_transcript.txt",
            "analyzed_at": get_bangkok_timestamp(),
    
        # Layer 1 metrics 
            "metrics": {
                "word_count": layer1_metrics["word_count"],
                "focus_score": layer1_metrics["focus_score"],
                "speaking_rate_wpm": layer1_metrics["speaking_rate_wpm"],
                "wpm_category": layer1_metrics["wpm_category"],
                "silence_ratio_percent": layer1_metrics["silence_ratio_percent"],
                "total_pause_seconds": layer1_metrics["total_pause_seconds"],
                "num_pauses": layer1_metrics["num_pauses"],
                "filler_density_percent": layer1_metrics["filler_density_percent"],
                "duration_seconds": layer1_metrics["duration_seconds"],
                "flagged_as_bad": layer1_metrics["flagged_as_bad"],
                "flag_reasons": layer1_metrics["flag_reasons"]
            },
    
            # Layer 2 AI evaluation
            "ai_evaluation": {
                "priority": layer2_result.get("priority", "UNKNOWN"),
                "reason": layer2_result.get("reason", "Không có đánh giá"),
                "content_score": layer2_result.get("content_score", 0),
                "communication_score": layer2_result.get("communication_score", 0),
                "overall_impression": layer2_result.get("overall_impression", "neutral"),
                "ai_available": layer2_result.get("ai_available", False)
            }
        }

        await update_metadata(folder_path, question_data={
            "index": question_index,
            **analysis_summary  # Merge tất cả fields vào question data
        })

        logger.info(f"✅ [Background] Full analysis completed for Q{question_index}")

        # Cleanup
        audio_path.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"❌ [Background] Analysis error Q{question_index}: {e}")
        try:
            await update_metadata(folder_path, question_data={
                "index": question_index,
                "transcriptionStatus": "failed",
                "analysisError": str(e)
            })
        except:
            pass

