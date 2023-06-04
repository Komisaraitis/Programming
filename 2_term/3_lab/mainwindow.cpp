#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPainter>
#include <QMouseEvent>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    for (int i=0; i< contentsRect().width(); i++) {
        for (int j=0; j< contentsRect().height(); j++) {
            painter.drawRect(rect.translated(i*HW,j*HW));
        }
    }
    if (!derevo.isNull())
        painter.drawEllipse(derevo, 100,100);
    painter.setBrush(QColor(110,40,60,100));
    if (!muravei.isNull())
        painter.drawEllipse(muravei, 10,10);
    if (!muha.isNull())
        painter.drawEllipse(muha, 10,10);
    QPen pen(QColor(180,90,180),4);
    painter.setPen(pen);
    for(int i=0;i<line.size();i++)
        painter.drawLine(line[i]);
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    k++;
    if(k==1) {
        derevo = event->pos();
    }
    if(k==2) {
        muravei=na_setke(event->pos());
    }
    if(k>=3) {
        line.clear();
        muha=na_setke(event->pos());
        QPoint move_muravei = muravei;
        while(move_muravei != muha) {
            QPoint move[4];
            move[0] = move_muravei + QPoint(0,HW);
            move[1] = move_muravei - QPoint(0,HW);
            move[2] = move_muravei + QPoint(HW,0);
            move[3] = move_muravei - QPoint(HW,0);
            QPoint p;
            int minn=100000;
            for(int i=0;i<4;i++){
                QPoint md=move[i]-derevo;
                int dl=QPoint::dotProduct(md,md);
                QPoint mm=move[i]-muha;
                int dl1=QPoint::dotProduct(mm,mm);
                if(dl>100*100 && move[i]!=move_muravei && dl1<minn ) {
                    minn = dl1;
                    p=move[i];
                }
            }

            line.append(QLine(move_muravei,p));
            move_muravei=p;



        }
        line.append(QLine(move_muravei,muha));

    }
    repaint();
}

QPoint MainWindow::na_setke(QPoint p)
{
    QPoint tochka;
    int minn=100000;
    for (int i=0; i< contentsRect().width(); i+=HW) {
        for (int j=0; j< contentsRect().height(); j+=HW) {
            QPoint point_setka = QPoint(0,0) + QPoint(i,j);
            QPoint tp = p - point_setka;
            int dlina=QPoint::dotProduct(tp,tp);
            if (dlina<minn){
                minn=dlina;
                tochka = point_setka;
            }
        }
    }
    return tochka;

}

