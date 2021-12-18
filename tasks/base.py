class base_problem(object):
    def __init__ (self):
        pass

    def name(self):
        pass

    def init_full(self, seed, tbs, db, hist):
        pass

    def full(self):
        pass

    def full_status(self, t):
        pass

    def full_result(self):
        pass

    def full_hist(self):
        pass

    def init_test(self, seed, tbs, db, hist, timer):
        pass

    def test(self):
        pass

    def test_status(self, t):
        pass

    def test_result(self):
        pass

    def test_hist(self):
        pass

    def init_generate(self, object, seed, tbs):
        pass

    def generate_name(self):
        pass

    def generate(self):
        pass

    def generate_status(self, i):
        pass

    def generate_save(self, path):
        pass

