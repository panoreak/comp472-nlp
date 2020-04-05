class Eval:
    def __init__(self, trace_file, eval_file):
        self._trace_file = open(trace_file, 'r', encoding="utf-8")
        self._eval_file = open(eval_file, 'w', encoding="utf-8")
        self._most_likely_correct = self.compute_most_likely_correct()
        self._accuracy = self.compute_accuracy()
        self._precision = self.compute_precision()
        self._recall = self.compute_recall()
        self._f1 = self.compute_f1()
        self._macro_and_weighted_f1 = self.compute_macro_and_weighted_f1()
        self._trace_file.close()

    def compute_most_likely_correct(self):
        most_likely_correct = {
            'eu': 0, 'ca': 0, 'gl': 0, 'es': 0, 'en': 0, 'pt': 0}
        self._trace_file.seek(0)
        for line in self._trace_file:
            if line.split()[-1] == 'correct':
                most_likely_correct[line.split()[1]] += 1
        return most_likely_correct

    def compute_predicted_total(self):
        most_likely_total = {
            'eu': 0, 'ca': 0, 'gl': 0, 'es': 0, 'en': 0, 'pt': 0}
        self._trace_file.seek(0)
        for line in self._trace_file:
            most_likely_total[line.split()[1]] += 1
        return most_likely_total

    def compute_actual_total(self):
        actual_total = {
            'eu': 0, 'ca': 0, 'gl': 0, 'es': 0, 'en': 0, 'pt': 0}
        self._trace_file.seek(0)
        for line in self._trace_file:
            actual_total[line.split()[-2]] += 1
        return actual_total

    def compute_accuracy(self):
        tweet_count = 0
        correct_count = 0
        self._trace_file.seek(0)
        for line in self._trace_file:
            correct_count += 1 if line.split()[-1] == 'correct' else 0
            tweet_count += 1
        return correct_count / tweet_count if tweet_count != 0 else 0.0

    def compute_precision(self):
        result = []
        predicted_total = self.compute_predicted_total()
        for key in predicted_total:
            p = self._most_likely_correct[key] / predicted_total[key]
            result.append(p)
        return result

    def compute_recall(self):
        result = []
        actual_total = self.compute_actual_total()
        for key in actual_total:
            r = self._most_likely_correct[key] / actual_total[key]
            result.append(r)
        return result

    def compute_f1(self):
        result = []
        for i in range(len(self._precision)):
            # TODO: handle division by zero cuz gl never predicts and gets correct
            if self._precision[i] + self._recall[i] == 0:
                result.append(0.0)
                continue
            f1 = 2 * (self._precision[i] * self._recall[i]) / \
                (self._precision[i] + self._recall[i])
            result.append(f1)
        return result

    def compute_macro_and_weighted_f1(self):
        result = [0] * 2
        avg = 0
        w_avg = 0
        actual_total = self.compute_actual_total()
        total_freq = 0

        for i, key in enumerate(actual_total):
            avg += self._f1[i]
            w_avg += actual_total[key] * self._f1[i]
            total_freq += actual_total[key]
        avg = avg / len(self._f1) if len(self._f1) != 0 else 0.0
        w_avg = w_avg / total_freq if total_freq != 0 else 0.0
        result[0] = avg
        result[1] = w_avg
        return result

    def write_to_file(self):
        self._eval_file.write('{}\r'.format(self._accuracy))
        for p in self._precision:
            self._eval_file.write('{}  '.format(p))
        self._eval_file.write('\r')
        for r in self._recall:
            self._eval_file.write('{}  '.format(r))
        self._eval_file.write('\r')
        for f in self._f1:
            self._eval_file.write('{}  '.format(f))
        self._eval_file.write('\r')
        self._eval_file.write('{}  {}'.format(
            self._macro_and_weighted_f1[0], self._macro_and_weighted_f1[1]))
        self._eval_file.close()
