# day_30 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

# def main():
#     capital = 100
#     profit = 0.020
#     profit30 = 0.040   
#     days = 365


#     print(f"{'Day':>5} | {'Capital ($)':>15}")
#     print("-" * 25)

#     for i in range(1, days + 1):

#         if i % 30 == 0:
#             capital += 50 

#         if i in day_30:
#             capital *= (1 + profit30) 

#         else:
#             capital *= (1 + profit)
#         print(f"{i:>5} | {capital:>15.2f}$")
#     print("-" * 34)
#     print(f"FINAL: {capital:.2f}$")
#     print()


# main()


import os
from pathlib import Path

def print_tree(directory, prefix=""):
    """
    Ин функсия сохтори файлҳоро ба таври рекурсивӣ чоп мекунад.
    """
    # Рӯйхати папкаҳое, ки бояд нодида гирифта шаванд (Blacklist)
    IGNORE_DIRS = {'.venv', '.git', '__pycache__', '.env','Logs','.idea', '.vscode', '.DS_Store'}
    
    # Гирифтани рӯйхати файлҳо ва папкаҳо
    try:
        path_obj = Path(directory)
        # Филтр кардан: venv ва дигар папкаҳои зиёдатиро хориҷ мекунем
        entries = [
            e for e in path_obj.iterdir() 
            if e.name not in IGNORE_DIRS
        ]
        
        # Сорт кардан: Аввал папкаҳо, баъд файлҳо (ё алифбоӣ)
        # Ин ҷо мо оддӣ алифбоӣ сорт мекунем
        entries.sort(key=lambda x: x.name.lower())
        
    except PermissionError:
        return

    # Шумораи умумии файлҳо дар ин папка
    count = len(entries)
    
    for i, entry in enumerate(entries):
        # Санҷиш: оё ин охирин файл дар рӯйхат аст?
        is_last = (i == count - 1)
        
        # Интихоби аломати дуруст
        connector = "└── " if is_last else "├── "
        
        # Чоп кардани ном
        # Агар папка бошад, дар охираш "/" мегузорем
        print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        
        # Агар ин папка бошад, дохили онро низ чоп мекунем (Recursion)
        if entry.is_dir():
            # Барои сатҳи оянда фосила ё хатти рост илова мекунем
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(entry, new_prefix)

if __name__ == "__main__":
    # Папкаи ҷорӣ (current directory)
    root_dir = "."
    
    # Номи папкаи асосиро чоп мекунем
    root_name = os.path.basename(os.path.abspath(root_dir))
    print(f"{root_name}/")
    
    # Оғози сохтани дарахт
    print_tree(root_dir)