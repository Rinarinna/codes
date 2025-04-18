var1 = input("Введите число 1: ")
var2 = input("Введите число 2: ")
print(var1, var2)

for i in range(1,8,2):
  print(i)

  var1=int(input("Первое число: "))
var2=int(input("Второе число: "))
print("Сумма: ",var1+var2)
print("Разность: ",var1-var2)
print("Произведение: ", var1*var2)
if(var2!=0):
  print("Деление: ", var1/var2)
else:
  print("Деление на 0")
print("В степень: ", var1**var2)

n = int(input("Введите число n: "))
for i in range(n+1):
  print(i)

  n = int(input("Введите число n: "))
if(n>0):
  for i in range(n+1):
    if i%2!=0:
     print(i)
else:
  print("n<=0")

  mas = input("Введите текст: ")
sym_f = input("Введите символ: ")
count = 0
print(mas)
for i in mas:
  if i == sym_f:
    count+=1
print(count)

mas = "1)Рассчет суммы двух вводимых числе\n2)Рассчет деления двух чисел\n3)Поиск количества символов в строке\nДля выхода введите \"stop\""

while True:
  print (mas)
  var = input()
  if var == "stop":
    break
  elif var == "1":
    var1=int(input("Введите 1 число:" ))
    var2=int(input("Введите 2 число: "))
    print("Результат:",var1+var2)
  elif var == "2":
    var1=int(input("Введите 1 число:" ))
    var2=int(input("Введите 2 число: "))
    if var2==0:
      print("деление на 0")
    else:
      print("Результат:",var1/var2)
  elif var == "3":
    string = input("Введите текст: ")
    sym_f = input("Введите символ:")
    print(string)
    count = 0
    for i in string:
      if i == sym_f:
        count+=1
    print("Символов:",count)

    
