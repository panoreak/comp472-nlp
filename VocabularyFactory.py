from abc import ABC, abstractmethod


class VocabularyFactory:
    @staticmethod
    def get_vocabulary(vocabulary_type):
        if vocabulary_type == '0':
            return CaseInsensitiveAlphabetChars()
        if vocabulary_type == '1':
            return CaseSensitiveAlphabetChars()
        if vocabulary_type == '2':
            return IsAlphaChars()


class Vocabulary(ABC):
    @abstractmethod
    def is_in_vocabulary(self, char):
        pass

    @abstractmethod
    def get_codepoint_list(self):
        pass

    @abstractmethod
    def get_size(self):
        pass


class CaseInsensitiveAlphabetChars(Vocabulary):
    def is_in_vocabulary(self, char):
        codepoint = ord(char)
        return 97 <= codepoint <= 122

    def get_codepoint_list(self):
        return list(range(97, 123))

    def get_size(self):
        return 26


class CaseSensitiveAlphabetChars(Vocabulary):
    def is_in_vocabulary(self, char):
        codepoint = ord(char)
        return 65 <= codepoint <= 90 or 97 <= codepoint <= 122

    def get_codepoint_list(self):
        codepoint_list = list(range(65, 91)) + (list(range(97, 123)))
        return codepoint_list

    def get_size(self):
        return 52  # 26*2


class IsAlphaChars(Vocabulary):
    def is_in_vocabulary(self, char):
        return char.isalpha()

    def get_codepoint_list(self):
        codepoint_list = []
        for codepoint in range(17 * 2 ** 16):
            ch = chr(codepoint)
            if ch.isalpha():
                codepoint_list.append(ch)
        return codepoint_list

    def get_size(self):
        return 116766  # From Project 2 FAQ
