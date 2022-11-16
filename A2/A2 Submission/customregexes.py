import re

def findUsernames(sentence):
  return re.findall("[^\wÀ-ÖØ-öø-ÿ@_]?(@[A-Za-z0-9_]{1,15})\\b", sentence)

def findUsernameCount(sentence):
  return len(findUsernames(sentence))

def findURLsandHTML(sentence):
  falsePositiveIndicators = ['but', 'don', 'we', 'what', 'you', 'night', 'since', 'especially', 'keep', 'lol', 'and', 'last']
  falsePositiveIndicatorsRegex = re.compile(r'^(' + r'|'.join(falsePositiveIndicators) + r')$', re.IGNORECASE)
  urls = re.findall("(?:http://|https://)?[A-Za-z0-9_]+\.[a-z][A-Za-z0-9_]{1,}[\.A-Za-z0-9_]*[/?[A-Za-z0-9_~]*]*\.?[A-Za-z0-9_]*\\b", sentence)
  all_urls = []
  for url in urls:
    dummy = re.split("\.", url)
    shouldAdd = True
    for comps in dummy:
      if (re.search(falsePositiveIndicatorsRegex, comps) or re.search("^[0-9_]*$", comps)):
        shouldAdd = False
    if shouldAdd:
      all_urls.append(url)
  if re.search("&quot;", sentence):
    all_urls.append("&quot;")
  if re.search("&amp;", sentence):
    all_urls.append("&amp;")
  if re.search("&lt;", sentence):
    all_urls.append("&lt;")
  return all_urls

def findURLs(sentence):
  falsePositiveIndicators = ['but', 'don', 'we', 'what', 'you', 'night', 'since', 'especially', 'keep', 'lol', 'and', 'last']
  falsePositiveIndicatorsRegex = re.compile(r'^(' + r'|'.join(falsePositiveIndicators) + r')$', re.IGNORECASE)
  urls = re.findall("(?:http://|https://)?[A-Za-z0-9_]+\.[a-z][A-Za-z0-9_]{1,}[\.A-Za-z0-9_]*[/?[A-Za-z0-9_~]*]*\.?[A-Za-z0-9_]*\\b", sentence)
  all_urls = []
  for url in urls:
    dummy = re.split("\.", url)
    shouldAdd = True
    for comps in dummy:
      if (re.search(falsePositiveIndicatorsRegex, comps) or re.search("^[0-9_]*$", comps)):
        shouldAdd = False
    if shouldAdd:
      all_urls.append(url)
  return all_urls

def findURLCount(sentence):
  return len(findURLs(sentence))

def lrstrip(text):
    return re.sub(r'^\s+|\s+$', '', text)

def match_empty(text):
    return re.match(r'^\s*$', text)

def findSentence(paragraph):
  # Split on ! and ? first
  sentences = re.split("[!?]+", paragraph)

  # Split on . followed by space and capital letter
  list_of_abbreviations = ['Mr', 'Ms', 'Mrs', "Dr"]
  new_sentence_components = []

  for sentence in sentences:
    if not re.search("\.\s+[A-Za-z ]", sentence):
      new_sentence_components.append(sentence)
    else:
      new_sentence_components.extend(re.split("\.\s+([A-Za-z ]+\w*)", sentence))
  sentences = new_sentence_components

  list_of_abbreviations = ['Mr', 'Ms', 'Mrs', "Dr", "Miss"]
  check_if_token_ends_with_abbreviation = re.compile(r'\s(' + r'|'.join(list_of_abbreviations) + r')')

  # Rejoin sentences that end with an abbreviation
  new_sentence_components = [sentences[0]]
  i = 0
  while i < len(sentences) - 1:
    if re.search(check_if_token_ends_with_abbreviation, sentences[i]):
      new_sentence_components[-1] = new_sentence_components[-1] + '. ' + sentences[i + 1]
      i += 1
    else:
      new_sentence_components.append(sentences[i + 1])
      i += 1
  sentences = new_sentence_components

  # Filter empty strings
  new_sentence_components = []
  for i in range(len(sentences)):
    if not match_empty(sentences[i]):
      new_sentence_components.append(sentences[i])
  sentences = new_sentence_components

  # Remove leading and trailing spaces
  for i in range(len(sentences)):
    sentences[i] = lrstrip(sentences[i])
  
  return (sentences)

def findSentenceCount(paragraph):
  return len(findSentence(paragraph))

def findTokens(sentence):
  # Split on Whitespace using regex
  tokens = re.split("\s+", sentence)

  # Check URLs and Users
  allTokens = []

  for token in tokens:

    if findURLs(token):
      URLs = findURLs(token)
      for url in URLs:
        indexes = re.finditer(url, token)
        for index in indexes:
          allTokens.append(token[index.start():index.end()])

    elif findUsernames(token):
      usernames = findUsernames(token)
      for username in usernames:
        indexes = re.finditer(username, token)
        for index in indexes:
          allTokens.append(token[index.start():index.end()])

    else:
      # split on punctuation
      if re.search("[\.!?]+", token):
        allTokens.extend(re.split("([\.!?]+)", token))
      else:
        allTokens.append(token)

  # Remove empty strings
  new_token_components = []
  for i in range(len(allTokens)):
    if not match_empty(allTokens[i]):
      new_token_components.append(allTokens[i])

  # Remove leading and trailing spaces
  allTokens = new_token_components
  for i in range(len(allTokens)):
    allTokens[i] = lrstrip(allTokens[i])
  return allTokens

def findTokenCount(sentence):
  return len(findTokens(sentence))

def findWordsStartingWithVowel(sentence):
  return re.findall("\\b[aeiouAEIOU][\w|À-Ö|Ø-ö|ø-ÿ|'|-]*\\b", sentence)

def countWordsStartingWithVowel(sentence):
  return len(findWordsStartingWithVowel(sentence))

def findWordsStartingWithConsonant(sentence):
  return re.findall("\\s([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ][\w|À-Ö|Ø-ö|ø-ÿ|'|-]*)\\b", sentence)

def countWordsStartingWithConsonant(sentence):
  return len(findWordsStartingWithConsonant(sentence))

def lowercase(text):
  lowercase = "abcdefghijklmnopqrstuvwxyz"
  uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  for i in range(26):
      text = re.sub(uppercase[i], lowercase[i], text)
  return text

def getDay(sentence):
  return re.findall("Fri|Mon|Thu|Sun|Tue|Wed|Sat", sentence)[0]