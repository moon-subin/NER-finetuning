from src.rules.regex_postrules import _label_threshold
ths = {"DATE":0.75,"default":0.5}
print("_label_threshold('B-DATE') =>", _label_threshold("B-DATE", ths))
print("_label_threshold('I-DATE') =>", _label_threshold("I-DATE", ths))
print("_label_threshold('B-TIME') =>", _label_threshold("B-TIME", {"default":0.6}))