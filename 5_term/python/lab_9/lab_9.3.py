import openpyxl
from openpyxl.chart import PieChart, Reference

file = "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_9\\salary.xlsx"
wb = openpyxl.load_workbook(file)
ws = wb.active

pie_chart = PieChart()
pie_chart.title = "Распределение зарплат по отделам"

info = Reference(ws, min_col=3, min_row=16, max_row=18)
title_info = Reference(ws, min_col=2, min_row=16, max_row=18)

pie_chart.add_data(info)
pie_chart.set_categories(title_info)

ws.add_chart(
    pie_chart, "L1"
)  

wb.save(file)


