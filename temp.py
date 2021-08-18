_list = [
"blue blue",
"red red",
"yellow yellow",
"dark dark"
]

new_list = []
for colors in _list: 
  s = colors.split()[0]
  new_list.append(f'{colors} {s}')

print(new_list)