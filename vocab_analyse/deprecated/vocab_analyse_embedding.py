import re
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QListWidget, QVBoxLayout, QLabel, QDialog, \
    QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox, QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, \
    QHeaderView, QPlainTextEdit
from sklearn.metrics.pairwise import cosine_similarity

from embedding import Embedding


class ContextEditDialog(QDialog):
    """编辑语境的对话框"""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑语境")

        self.edit = QPlainTextEdit(text)

        layout = QFormLayout()
        layout.addRow("语境内容：", self.edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_data(self):
        return self.edit.toPlainText()


class WordEditDialog(QDialog):
    """编辑词语的对话框"""

    def __init__(self, contexts, word_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑词语")

        self.word_edit = QLineEdit()
        self.context_combo = QComboBox()
        self.match_index_spin = QSpinBox()
        self.match_index_spin.setMinimum(1)
        self.match_index_spin.setMaximum(9999)

        # 填充语境列表
        self.context_combo.addItem(f"{0}. None")
        for i, ctx in enumerate(contexts):
            self.context_combo.addItem(f"{i + 1}. {ctx}")

        layout = QFormLayout()
        layout.addRow("词语：", self.word_edit)
        layout.addRow("所属语境：", self.context_combo)
        layout.addRow("第 n 个匹配词：", self.match_index_spin)

        if word_data is not None:
            self.word_edit.setText(word_data["word"])
            self.match_index_spin.setValue(word_data["match_index"])
            if 0 <= word_data["context_index"] <= len(contexts):
                self.context_combo.setCurrentIndex(word_data["context_index"])

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_data(self):
        return {
            "word": self.word_edit.text(),
            "context_index": self.context_combo.currentIndex(),
            "match_index": self.match_index_spin.value(),
        }


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.emb = Embedding("BAAI/bge-m3")

        self.setWindowTitle("词语相似度计算工具")
        self.setGeometry(100, 100, 800, 600)

        layout_main = QVBoxLayout()
        layout_interface = QHBoxLayout()

        # Layout Interface - Context
        self.context_list = QListWidget()
        self.context_list.itemDoubleClicked.connect(self.edit_context)

        context_button = QPushButton("+")
        context_button.clicked.connect(self.add_context)

        layout_context = QVBoxLayout()
        layout_context.addWidget(QLabel("语境列表"))
        layout_context.addWidget(self.context_list)
        layout_context.addWidget(context_button)
        layout_interface.addLayout(layout_context)

        # Layout Interface - Word
        self.word_table = QTableWidget(0, 4)
        self.word_table.setHorizontalHeaderLabels(["词语", "所属语境", "第n个匹配词", "选中"])
        self.word_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.word_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.word_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.word_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.word_table.cellDoubleClicked.connect(self.edit_word)

        word_button = QPushButton("+")
        word_button.clicked.connect(self.add_word)

        layout_word = QVBoxLayout()
        layout_word.addWidget(QLabel("词语列表"))
        layout_word.addWidget(self.word_table)
        layout_word.addWidget(word_button)
        layout_interface.addLayout(layout_word)

        # Layout Interface - Calculator
        self.expr1_edit = QLineEdit()
        self.expr2_edit = QLineEdit()
        self.expr1_edit.setPlaceholderText("vn代表第n行词的向量。")
        self.expr2_edit.setPlaceholderText("例如: v3 - v4")
        calc_sim_button = QPushButton("计算相似度")
        calc_sim_button.clicked.connect(self.calc_similarity)

        layout_calculator = QVBoxLayout()
        layout_calculator.addWidget(self.expr1_edit)
        layout_calculator.addWidget(self.expr2_edit)
        layout_calculator.addWidget(calc_sim_button)
        layout_interface.addLayout(layout_calculator)

        layout_main.addLayout(layout_interface)

        # Result
        self.result_output = QPlainTextEdit()
        self.result_output.setReadOnly(True)
        layout_main.addWidget(self.result_output)

        self.setLayout(layout_main)

    def get_word_vector_by_row(self, row):
        if row < 0 or row >= self.word_table.rowCount():
            raise ValueError(f"词条索引越界：{row + 1}")

        item = self.word_table.item(row, 0)
        if item is None:
            raise ValueError(f"第 {row + 1} 行为空")

        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            raise ValueError(f"第 {row + 1} 行没有词语数据")

        if data["context_index"] == 0:
            return self.emb.get_vector(data["word"]).reshape(1, -1)

        word = data["word"]
        context_row = data["context_index"] - 1
        if context_row < 0 or context_row >= self.context_list.count():
            raise ValueError(f"第 {row + 1} 行引用的语境不存在")

        context = self.context_list.item(context_row).text()

        start = -1
        for _ in range(data["match_index"]):
            start = context.find(word, start + 1)
            if start == -1:
                raise ValueError(f"未在语境中找到第 {data['match_index']} 个“{word}”")

        end = start + len(word)
        return self.emb.get_vector(context, (start, end)).reshape(1, -1)

    def evaluate_vector_expr(self, expr):
        expr = expr.strip()
        if not expr:
            raise ValueError("公式不能为空")
        if not re.fullmatch(r"[\d\sv+\-*/().]+", expr):
            raise ValueError("公式中包含非法字符")

        refs = list(set(re.findall(r'\bv\d+\b', expr)))
        if not refs:
            raise ValueError("公式中没有引用任何向量，例如 v1-v2")

        # 准备允许使用的变量
        allowed_names = {}

        for ref in refs:
            index = int(ref[1:])
            row = index - 1
            vec = self.get_word_vector_by_row(row)
            allowed_names[f"v{index}"] = vec

        try:
            result = eval(expr, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"公式计算失败: {e}")

        return result

    def calc_similarity(self):
        if not self.expr1_edit.text().strip():
            QMessageBox.warning(self, "提示", "向量1未输入计算方式")
            return
        if not self.expr2_edit.text().strip():
            QMessageBox.warning(self, "提示", "向量2未输入计算方式")
            return

        vec1 = self.evaluate_vector_expr(self.expr1_edit.text().strip())
        vec2 = self.evaluate_vector_expr(self.expr2_edit.text().strip())

        similarity = cosine_similarity(vec1, vec2)

        self.result_output.setPlainText(f"相似度为：{similarity}")

    def get_contexts(self):
        return [self.context_list.item(i).text() for i in range(self.context_list.count())]

    def add_context(self):
        self.context_list.addItem("Empty")

    def add_word(self):
        row = self.word_table.rowCount()
        self.word_table.insertRow(row)

        self.word_table.setItem(row, 0, QTableWidgetItem("Empty"))
        self.word_table.setItem(row, 1, QTableWidgetItem("0"))
        self.word_table.setItem(row, 2, QTableWidgetItem("1"))
        self.word_table.setItem(row, 3, QTableWidgetItem(""))

        self.word_table.item(row, 0).setData(
            Qt.ItemDataRole.UserRole,
            {
                "word": "Empty",
                "context_index": 0,
                "match_index": 1
            }
        )

    def edit_context(self, item):
        old_text = item.text()

        dialog = ContextEditDialog(old_text, self)
        if dialog.exec():
            new_text = dialog.get_data()
            if not new_text:
                QMessageBox.warning(self, "提示", "语境不能为空")
                return
            item.setText(new_text)

    def edit_word(self, row, column):
        item = self.word_table.item(row, 0)
        old_data = item.data(Qt.ItemDataRole.UserRole)

        dialog = WordEditDialog(self.get_contexts(), old_data, self)
        if dialog.exec():
            new_data = dialog.get_data()
            if not new_data["word"]:
                QMessageBox.warning(self, "提示", "词语不能为空")
                return
            self.word_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, new_data)

            item.setText(new_data["word"])
            self.word_table.item(row, 1).setText(str(new_data["context_index"]))
            self.word_table.item(row, 2).setText(str(new_data["match_index"]))


def main():
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec()


if __name__ == "__main__":
    main()
