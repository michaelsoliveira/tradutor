import pytesseract
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
import pyttsx3
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
import speech_recognition as sr
import math
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configuração do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ajuste o caminho no Windows ou deixe padrão no Linux/Mac


if torch.cuda.is_available():  
    dev = "cuda"
else:  
    dev = "cpu" 
device = torch.device(dev)

# Inicializar o motor de fala pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('voice', 'brazil')
engine.setProperty('volume', 1.0)

# Dicionário de idiomas suportados (mantido como está)
idiomas = {
    'Afrikaans': 'af', 'Árabe': 'ar', 'Bengali': 'bn', 'Cantonês': 'yue', 'Catalão': 'ca',
    'Chinês': 'zh-tw', 'Croata': 'hr', 'Checo': 'cs', 'Dinamarquês': 'da', 'Holandês': 'nl',
    'Inglês': 'en', 'Filipino': 'fil', 'Finlandês': 'fi', 'Francês': 'fr', 'Alemão': 'de',
    'Grego': 'el', 'Gujarati': 'gu', 'Hebraico': 'he', 'Hindi': 'hi', 'Húngaro': 'hu',
    'Indonésio': 'id', 'Italiano': 'it', 'Japonês': 'ja', 'Javanês': 'jw', 'Coreano': 'ko',
    'Letão': 'lv', 'Lituano': 'lt', 'Malaio': 'ms', 'Marata': 'mr', 'Norueguês': 'no',
    'Polonês': 'pl', 'Português Brasil': 'pt', 'Português': 'pt',
    'Romeno': 'ro', 'Russo': 'ru', 'Sérvio': 'sr', 'Eslovaco': 'sk', 'Esloveno': 'sl',
    'Espanhol': 'es', 'Suaíli': 'sw', 'Sueco': 'sv', 'Tâmil': 'ta', 'Telugu': 'te',
    'Tailandês': 'th', 'Turco': 'tr', 'Ucraniano': 'uk', 'Vietnamita': 'vi', 'Galês': 'cy'
}

def falar(texto):
    engine.say(texto)
    engine.runAndWait()

def reconhecer_comando(timeout=3):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        print("Ouvindo...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            comando = recognizer.recognize_google(audio, language='pt-BR')
            print(f"Comando reconhecido: {comando}")
            return comando.lower()
        except:
            return None

def select_frame_por_voz(cap, result):
    while True:
        ret, frame = cap.read()
        if not ret:
            falar("Estou com dificuldades para capturar a imagem. Podemos tentar de novo?")
            continue
        cv2.imshow('frame', frame)

        if result['comando']:
            comando = result['comando']
            result['comando'] = None
            if "capturar" in comando or "enter" in comando:
                falar("Certo, vou capturar esta imagem.")
                return frame
            elif "sair" in comando or "finalizar" in comando:
                falar("Você quer que eu encerre o programa?")
                if reconhecer_comando() in ['sim', 'pode', 'ok', 'encerre']:
                    result['fim'] = True
                    return None
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return None

def preprocess_image(image):
    return cv2.resize(image, (128, 128)) / 255.0
     
# Função para realizar OCR na imagem
def extract_text_from_image(image_path):
    try:
        # image = Image.open(image_path)
        gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)  # OCR para texto em inglês
        return text.strip()
    except Exception as e:
        return f"Erro ao processar a imagem: {e}"

def selecionar_idioma_por_voz():
    while True:
        falar("Por favor, me diga para qual idioma você gostaria de traduzir.")
        comando = reconhecer_comando()
        if comando:
            for nome, codigo in idiomas.items():
                if nome.lower() in comando:
                    falar(f"Entendi que você quer traduzir para {nome}")
                    return codigo
        falar("Desculpe, não consegui identificar o idioma. Vamos tentar novamente?")

# Função para traduzir texto usando um modelo pré-treinado
def translate_text(text, source_lang="en", target_lang="pt"):
    try:
        model_name = f"Helsinki-NLP/opus-mt-tc-big-{source_lang}-{target_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        model.to(device)
        lt = LineTokenizer()
        batch_size = 8

        paragraphs = lt.tokenize(text)   
        translated_paragraphs = []

        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            batches = math.ceil(len(sentences) / batch_size)     
            translated = []
            for i in range(batches):
                sent_batch = sentences[i*batch_size:(i+1)*batch_size]
                model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    translated_batch = model.generate(**model_inputs)
                translated += translated_batch
            translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            translated_paragraphs += [" ".join(translated)]

        translated_text = "\n".join(translated_paragraphs)
        
        return translated_text
    except Exception as e:
        return f"Erro na tradução: {e}"

def obter_comandos_de_voz(result):
    while not result['fim']:
        comando = reconhecer_comando()
        if comando:
            result['comando'] = comando
    
def main():
    falar("Olá! Bem-vindo ao SICOMUV, seu assistente de comunicação e tradução. Como posso te ajudar hoje?")

    idioma_selecionado = selecionar_idioma_por_voz()
    falar("Ótimo! Agora que escolhemos o idioma, você pode me pedir para capturar uma imagem ou encerrar o programa. O que você prefere?")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            falar("Estou com problemas para acessar a câmera. Pode verificar se ela está conectada corretamente?")
            return

    result = {'comando': None, 'fim': False}
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(obter_comandos_de_voz, result)

        while not result['fim']:
            falar("Estou pronto para capturar uma imagem. Diga 'capturar' quando quiser.")
            selected_frame = select_frame_por_voz(cap, result)
            if result['fim']:
                break
            if selected_frame is not None:
                falar("Imagem capturada! Estou processando, só um instante.")
                # Processamento
                print("Processando a imagem para extrair texto...")
                extracted_text = extract_text_from_image(selected_frame)
                result['comando'] = None
                print(f"Texto extraído: {extracted_text}")

                if extracted_text:
                    print("\nTraduzindo o texto...")
                    translated_text = translate_text(extracted_text, source_lang="en", target_lang=idioma_selecionado)
                    print(f"Texto traduzido: {translated_text}")

                    print("\nLendo o texto traduzido em voz...")
                    falar(translated_text)
                    result['fim'] = True
                # processar_imagem(selected_frame, model, idioma_selecionado)

    falar("Estou encerrando o programa. Foi um prazer ajudar você hoje. Obrigado por usar o SICOMUV!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
