from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout, 
                             QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter, QBrush, QRegion
from PyQt5.QtCore import Qt, QSize
import os
import logging

logger = logging.getLogger(__name__)

class Dashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DashboardWidget")
        self._init_ui()

    def _apply_shadow_effect(self, widget, blur_radius=20, x_offset=0, y_offset=5, color_alpha=50):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QColor(0, 0, 0, color_alpha)) 
        shadow.setOffset(x_offset, y_offset)
        widget.setGraphicsEffect(shadow)
        widget.setMask(QRegion(widget.rect()))


    def _init_ui(self):
        """Initialize the user interface for the dashboard."""
        
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        main_content_container_widget = QWidget()
        scroll_area.setWidget(main_content_container_widget)

        main_layout = QVBoxLayout(main_content_container_widget)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setSpacing(15)

        app_title_layout = QHBoxLayout()
        app_title_layout.setAlignment(Qt.AlignLeft)
        app_title_layout.setSpacing(15)

        icon_path_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icons")
        app_icon_path = os.path.join(icon_path_root, "ai.ico") 
        
        if os.path.exists(app_icon_path):
            icon_label = QLabel()
            pixmap = QPixmap(app_icon_path)
            icon_label.setPixmap(pixmap.scaled(QSize(50, 50), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            app_title_layout.addWidget(icon_label)

        title_label = QLabel("Machine Learning Studio")
        title_label.setObjectName("AppTitleLabel")
        title_label.setWordWrap(False)
        app_title_layout.addWidget(title_label)
        
        main_layout.addLayout(app_title_layout)

        project_info_group = QFrame()
        project_info_group.setObjectName("CardFrame")
        project_info_group_layout = QVBoxLayout(project_info_group)
        project_info_group_layout.setContentsMargins(20, 15, 20, 15)

        project_info_text = (
            "<div style='text-align:center;'>"
            "<span style='font-size:16pt; font-weight:bold; color:#2C3E50;'>Final Year Project - DAE CIT</span><br>"
            "<span style='font-size:12pt; color:#34495E;'>Developed by: <b style='color:#1ABC9C; font-weight:normal;'>Saeed Ur Rehman</b></span>"
            "</div>"
        )
        project_info_label = QLabel(project_info_text)
        project_info_label.setObjectName("ProjectInfoLabel")
        project_info_label.setAlignment(Qt.AlignCenter)
        project_info_label.setWordWrap(True)
        project_info_group_layout.addWidget(project_info_label)
        
        main_layout.addWidget(project_info_group)
        self._apply_shadow_effect(project_info_group, blur_radius=25, y_offset=5, color_alpha=40)

        separator = QFrame()
        separator.setObjectName("SubtleSeparatorLine")
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)
        main_layout.addSpacing(8)

        instructions_title_label = QLabel("Getting Started")
        instructions_title_label.setObjectName("InstructionsTitleLabel")
        instructions_title_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(instructions_title_label)
        
        welcome_message_text = (
            "<div style='color:#34495E; font-size:11pt;'>" 
            "This application provides an integrated environment for various machine learning workflows. "
            "Navigate using the sidebar to:"
            "<ul style='margin-left: 20px; margin-top: 8px; margin-bottom: 8px; line-height: 1.5;'>"
            "<li><b>Load and explore datasets</b> - Import CSV files, view statistics, and visualize distributions</li>"
            "<li><b>Clean and prepare data</b> - Handle missing values, encode categorical features, and scale numerical data</li>"
            "<li><b>Train machine learning models</b> - Classification, regression, clustering with various algorithms</li>"
            "<li><b>Evaluate model performance</b> - Accuracy, precision, recall, F1-score, ROC curves, and confusion matrices</li>"
            "</ul>"
            "To begin, please import your dataset using the <b style='color:#1ABC9C;'>'Import Data'</b> option on the sidebar."
            "</div>"
        )
        welcome_message_label = QLabel(welcome_message_text)
        welcome_message_label.setObjectName("WelcomeMessageCard") 
        welcome_message_label.setAlignment(Qt.AlignLeft | Qt.AlignTop) 
        welcome_message_label.setWordWrap(True)
        welcome_message_label.setMinimumWidth(400)

        main_layout.addWidget(welcome_message_label)
        self._apply_shadow_effect(welcome_message_label, blur_radius=30, y_offset=5, color_alpha=30)

        main_layout.addStretch(1) 
        
        dashboard_base_layout = QVBoxLayout(self)
        dashboard_base_layout.setContentsMargins(0,0,0,0)
        dashboard_base_layout.addWidget(scroll_area)

        self.setStyleSheet("""
            #DashboardWidget {
                background-color: #F8F9FA; 
            }
            QScrollArea {
                 border: none; 
                 background-color: transparent;
            }
            
            #AppTitleLabel {
                font-family: "Segoe UI", Arial, Helvetica, sans-serif; 
                font-size: 28px;
                font-weight: 600; 
                color: #2C3E50; 
                padding-top: 2px;
            }

            #CardFrame, #WelcomeMessageCard {
                background-color: #FFFFFF; 
                border-radius: 10px;
                border: 1px solid #EFF1F3; 
            }
            
            #WelcomeMessageCard { 
                 padding: 15px 25px;
            }

            #ProjectInfoLabel {
                font-family: "Segoe UI Light", Arial, Helvetica, sans-serif;
                line-height: 1.4;
            }

            #SubtleSeparatorLine {
                background-color: #E3E6E8; 
                border: none; 
                margin-top: 4px; 
                margin-bottom: 4px;
            }

            #InstructionsTitleLabel {
                font-family: "Segoe UI Semibold", Arial, Helvetica, sans-serif;
                font-size: 22px;
                font-weight: 600;
                color: #34495E; 
                margin-top: 8px;
                margin-bottom: 8px;
            }

            #WelcomeMessageLabel b { 
                 color: #1ABC9C; 
                 font-weight: 600; 
            }
        """)