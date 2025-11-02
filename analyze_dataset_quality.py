#!/usr/bin/env python3
"""
Анализ качества датасета PlantVillage
Скрипт проводит полный анализ качества данных и выводит отчет.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Библиотеки для работы с изображениями
try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Установка необходимых библиотек...")
    os.system("pip install pillow matplotlib seaborn -q")
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns

class DatasetQualityAnalyzer:
    """Класс для анализа качества датасета"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'classes': {},
            'statistics': {},
            'issues': [],
            'warnings': []
        }
        
    def analyze_structure(self):
        """Анализ структуры датасета"""
        print("📁 Анализ структуры датасета...")
        
        # Получаем все директории классов
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        self.results['total_classes'] = len(class_dirs)
        self.results['classes'] = {}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            files = [f for f in class_dir.iterdir() if f.is_file()]
            # Определяем изображения через расширения
            images = []
            for f in files:
                # Проверка по расширению
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    images.append(f)
                # Проверка на JPG/PNG без точки или с заглавными буквами
                elif any(f.name.upper().endswith(ext) for ext in ['.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', 'JPG', 'JPEG', 'PNG']):
                    images.append(f)
            
            self.results['classes'][class_name] = {
                'total_files': len(files),
                'image_files': len(images),
                'other_files': len(files) - len(images),
                'file_extensions': list(set([f.suffix.lower() if f.suffix else 'no_ext' for f in images]))
            }
        
        print(f"   ✅ Найдено классов: {self.results['total_classes']}")
        
    def analyze_images(self):
        """Анализ качества изображений"""
        print("\n🖼️  Анализ изображений...")
        
        image_stats = {
            'widths': [],
            'heights': [],
            'sizes_bytes': [],
            'formats': [],
            'corrupted': [],
            'total_images': 0,
            'classes_stats': {}
        }
        
        for class_name, class_info in self.results['classes'].items():
            class_dir = self.dataset_path / class_name
            # Находим все изображения в директории класса
            images = []
            for f in class_dir.iterdir():
                if f.is_file():
                    # Проверка по расширению
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        images.append(f)
                    # Проверка на JPG/PNG без точки или с заглавными буквами
                    elif any(f.name.upper().endswith(ext) for ext in ['.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', 'JPG', 'JPEG', 'PNG']):
                        images.append(f)
            
            class_image_stats = {
                'widths': [],
                'heights': [],
                'sizes': [],
                'formats': [],
                'corrupted_count': 0
            }
            
            print(f"   Обработка класса: {class_name} ({len(images)} изображений)...")
            
            for img_path in images[:len(images)]:  # Обрабатываем все изображения
                try:
                    # Размер файла
                    file_size = img_path.stat().st_size
                    image_stats['sizes_bytes'].append(file_size)
                    class_image_stats['sizes'].append(file_size)
                    
                    # Размеры изображения
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_stats['widths'].append(width)
                        image_stats['heights'].append(height)
                        image_stats['formats'].append(img.format)
                        
                        class_image_stats['widths'].append(width)
                        class_image_stats['heights'].append(height)
                        class_image_stats['formats'].append(img.format)
                        
                        image_stats['total_images'] += 1
                        
                except Exception as e:
                    image_stats['corrupted'].append(str(img_path))
                    class_image_stats['corrupted_count'] += 1
                    self.results['issues'].append({
                        'type': 'corrupted_image',
                        'file': str(img_path),
                        'error': str(e)
                    })
            
            # Статистика по классу
            if class_image_stats['widths']:
                self.results['classes'][class_name]['image_stats'] = {
                    'min_width': min(class_image_stats['widths']),
                    'max_width': max(class_image_stats['widths']),
                    'min_height': min(class_image_stats['heights']),
                    'max_height': max(class_image_stats['heights']),
                    'avg_width': np.mean(class_image_stats['widths']),
                    'avg_height': np.mean(class_image_stats['heights']),
                    'avg_size_mb': np.mean(class_image_stats['sizes']) / (1024 * 1024),
                    'total_size_mb': sum(class_image_stats['sizes']) / (1024 * 1024),
                    'format_distribution': dict(Counter(class_image_stats['formats'])),
                    'corrupted_count': class_image_stats['corrupted_count']
                }
        
        # Общая статистика
        if image_stats['widths']:
            self.results['statistics'] = {
                'total_images': image_stats['total_images'],
                'corrupted_images': len(image_stats['corrupted']),
                'width': {
                    'min': int(min(image_stats['widths'])),
                    'max': int(max(image_stats['widths'])),
                    'mean': float(np.mean(image_stats['widths'])),
                    'median': float(np.median(image_stats['widths'])),
                    'std': float(np.std(image_stats['widths']))
                },
                'height': {
                    'min': int(min(image_stats['heights'])),
                    'max': int(max(image_stats['heights'])),
                    'mean': float(np.mean(image_stats['heights'])),
                    'median': float(np.median(image_stats['heights'])),
                    'std': float(np.std(image_stats['heights']))
                },
                'file_size': {
                    'min_mb': float(min(image_stats['sizes_bytes']) / (1024 * 1024)),
                    'max_mb': float(max(image_stats['sizes_bytes']) / (1024 * 1024)),
                    'mean_mb': float(np.mean(image_stats['sizes_bytes']) / (1024 * 1024)),
                    'total_size_gb': float(sum(image_stats['sizes_bytes']) / (1024 * 1024 * 1024))
                },
                'format_distribution': dict(Counter(image_stats['formats']))
            }
        
        print(f"   ✅ Проанализировано изображений: {image_stats['total_images']}")
        if image_stats['corrupted']:
            print(f"   ⚠️  Поврежденных изображений: {len(image_stats['corrupted'])}")
    
    def analyze_class_balance(self):
        """Анализ баланса классов"""
        print("\n⚖️  Анализ баланса классов...")
        
        class_counts = {}
        for class_name, class_info in self.results['classes'].items():
            image_count = class_info.get('image_files', 0)
            class_counts[class_name] = image_count
        
        if class_counts:
            total = sum(class_counts.values())
            self.results['statistics']['class_balance'] = {
                'total_images': total,
                'class_distribution': class_counts,
                'min_class': min(class_counts.values()),
                'max_class': max(class_counts.values()),
                'imbalance_ratio': max(class_counts.values()) / min(class_counts.values()) if min(class_counts.values()) > 0 else 0,
                'class_counts_stats': {
                    'mean': float(np.mean(list(class_counts.values()))),
                    'std': float(np.std(list(class_counts.values()))),
                    'median': float(np.median(list(class_counts.values())))
                }
            }
            
            # Предупреждения о дисбалансе
            if self.results['statistics']['class_balance']['imbalance_ratio'] > 5:
                self.results['warnings'].append({
                    'type': 'class_imbalance',
                    'severity': 'high',
                    'message': f'Сильный дисбаланс классов: соотношение {self.results["statistics"]["class_balance"]["imbalance_ratio"]:.2f}:1'
                })
            elif self.results['statistics']['class_balance']['imbalance_ratio'] > 2:
                self.results['warnings'].append({
                    'type': 'class_imbalance',
                    'severity': 'medium',
                    'message': f'Умеренный дисбаланс классов: соотношение {self.results["statistics"]["class_balance"]["imbalance_ratio"]:.2f}:1'
                })
    
    def create_visualizations(self, output_dir='results'):
        """Создание визуализаций"""
        print("\n📊 Создание визуализаций...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Распределение классов
        if 'class_balance' in self.results['statistics']:
            plt.figure(figsize=(14, 8))
            class_counts = self.results['statistics']['class_balance']['class_distribution']
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            plt.subplot(2, 2, 1)
            plt.barh(classes, counts)
            plt.xlabel('Количество изображений')
            plt.title('Распределение классов')
            plt.tight_layout()
            
            plt.subplot(2, 2, 2)
            plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            plt.title('Процентное распределение классов')
            
            # 2. Распределение размеров изображений
            if 'width' in self.results['statistics']:
                plt.subplot(2, 2, 3)
                widths = [self.results['statistics']['width']['mean']] * 10
                heights = [self.results['statistics']['height']['mean']] * 10
                
                # Получаем реальные размеры для гистограммы
                all_widths = []
                all_heights = []
                for class_name, class_info in self.results['classes'].items():
                    if 'image_stats' in class_info:
                        # Примерные значения на основе статистики
                        mean_w = class_info['image_stats']['avg_width']
                        mean_h = class_info['image_stats']['avg_height']
                        count = class_info['image_files']
                        all_widths.extend([mean_w] * count)
                        all_heights.extend([mean_h] * count)
                
                if all_widths:
                    plt.hist(all_widths, bins=30, alpha=0.7, label='Ширина')
                    plt.hist(all_heights, bins=30, alpha=0.7, label='Высота')
                    plt.xlabel('Размеры (пиксели)')
                    plt.ylabel('Частота')
                    plt.title('Распределение размеров изображений')
                    plt.legend()
                
                # 3. Распределение размеров файлов
                plt.subplot(2, 2, 4)
                if 'class_balance' in self.results['statistics']:
                    sizes = []
                    for class_name, class_info in self.results['classes'].items():
                        if 'image_stats' in class_info:
                            avg_size = class_info['image_stats']['avg_size_mb']
                            count = class_info['image_files']
                            sizes.extend([avg_size] * count)
                    
                    if sizes:
                        plt.hist(sizes, bins=30, alpha=0.7)
                        plt.xlabel('Размер файла (MB)')
                        plt.ylabel('Частота')
                        plt.title('Распределение размеров файлов')
            
            plt.tight_layout()
            plt.savefig(output_path / 'dataset_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Графики сохранены: {output_path / 'dataset_analysis.png'}")
    
    def generate_report(self, output_dir='results'):
        """Генерация текстового отчета"""
        print("\n📝 Генерация отчета...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ОТЧЕТ ПО КАЧЕСТВУ ДАТАСЕТА PLANTVILLAGE")
        report_lines.append("=" * 80)
        report_lines.append(f"\nДата анализа: {self.results['timestamp']}")
        report_lines.append(f"Путь к датасету: {self.results['dataset_path']}")
        report_lines.append("")
        
        # Общая информация
        report_lines.append("=" * 80)
        report_lines.append("1. ОБЩАЯ ИНФОРМАЦИЯ")
        report_lines.append("=" * 80)
        report_lines.append(f"Всего классов: {self.results['total_classes']}")
        
        if 'statistics' in self.results:
            stats = self.results['statistics']
            report_lines.append(f"Всего изображений: {stats.get('total_images', 'N/A')}")
            report_lines.append(f"Поврежденных изображений: {stats.get('corrupted_images', 0)}")
            if 'file_size' in stats:
                report_lines.append(f"Общий размер датасета: {stats['file_size']['total_size_gb']:.2f} GB")
        
        # Статистика по размерам изображений
        if 'statistics' in self.results and 'width' in self.results['statistics']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("2. СТАТИСТИКА ПО РАЗМЕРАМ ИЗОБРАЖЕНИЙ")
            report_lines.append("=" * 80)
            
            stats = self.results['statistics']
            report_lines.append(f"\nШирина (пиксели):")
            report_lines.append(f"  Минимум: {stats['width']['min']}")
            report_lines.append(f"  Максимум: {stats['width']['max']}")
            report_lines.append(f"  Среднее: {stats['width']['mean']:.2f}")
            report_lines.append(f"  Медиана: {stats['width']['median']:.2f}")
            report_lines.append(f"  Стандартное отклонение: {stats['width']['std']:.2f}")
            
            report_lines.append(f"\nВысота (пиксели):")
            report_lines.append(f"  Минимум: {stats['height']['min']}")
            report_lines.append(f"  Максимум: {stats['height']['max']}")
            report_lines.append(f"  Среднее: {stats['height']['mean']:.2f}")
            report_lines.append(f"  Медиана: {stats['height']['median']:.2f}")
            report_lines.append(f"  Стандартное отклонение: {stats['height']['std']:.2f}")
        
        # Статистика по размерам файлов
        if 'statistics' in self.results and 'file_size' in self.results['statistics']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("3. СТАТИСТИКА ПО РАЗМЕРАМ ФАЙЛОВ")
            report_lines.append("=" * 80)
            
            stats = self.results['statistics']['file_size']
            report_lines.append(f"Минимальный размер: {stats['min_mb']:.2f} MB")
            report_lines.append(f"Максимальный размер: {stats['max_mb']:.2f} MB")
            report_lines.append(f"Средний размер: {stats['mean_mb']:.2f} MB")
        
        # Распределение форматов
        if 'statistics' in self.results and 'format_distribution' in self.results['statistics']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("4. РАСПРЕДЕЛЕНИЕ ФОРМАТОВ ФАЙЛОВ")
            report_lines.append("=" * 80)
            for fmt, count in self.results['statistics']['format_distribution'].items():
                report_lines.append(f"  {fmt}: {count} файлов")
        
        # Баланс классов
        if 'statistics' in self.results and 'class_balance' in self.results['statistics']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("5. БАЛАНС КЛАССОВ")
            report_lines.append("=" * 80)
            
            balance = self.results['statistics']['class_balance']
            report_lines.append(f"\nВсего изображений: {balance['total_images']}")
            report_lines.append(f"Минимум в классе: {balance['min_class']}")
            report_lines.append(f"Максимум в классе: {balance['max_class']}")
            report_lines.append(f"Соотношение дисбаланса: {balance['imbalance_ratio']:.2f}:1")
            report_lines.append(f"\nРаспределение по классам:")
            
            for class_name, count in sorted(balance['class_distribution'].items(), 
                                          key=lambda x: x[1], reverse=True):
                percentage = (count / balance['total_images']) * 100
                report_lines.append(f"  {class_name}: {count} ({percentage:.2f}%)")
            
            # Статистика
            stats_balance = balance['class_counts_stats']
            report_lines.append(f"\nСтатистика по классам:")
            report_lines.append(f"  Среднее: {stats_balance['mean']:.2f}")
            report_lines.append(f"  Медиана: {stats_balance['median']:.2f}")
            report_lines.append(f"  Стандартное отклонение: {stats_balance['std']:.2f}")
        
        # Детальная информация по классам
        report_lines.append("\n" + "=" * 80)
        report_lines.append("6. ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО КЛАССАМ")
        report_lines.append("=" * 80)
        
        for class_name, class_info in self.results['classes'].items():
            report_lines.append(f"\n📁 Класс: {class_name}")
            report_lines.append(f"  Изображений: {class_info['image_files']}")
            report_lines.append(f"  Других файлов: {class_info['other_files']}")
            
            if 'image_stats' in class_info:
                stats = class_info['image_stats']
                report_lines.append(f"  Размеры:")
                report_lines.append(f"    Ширина: {stats['min_width']:.0f} - {stats['max_width']:.0f} (среднее: {stats['avg_width']:.2f})")
                report_lines.append(f"    Высота: {stats['min_height']:.0f} - {stats['max_height']:.0f} (среднее: {stats['avg_height']:.2f})")
                report_lines.append(f"  Размер данных: {stats['total_size_mb']:.2f} MB")
                if stats['corrupted_count'] > 0:
                    report_lines.append(f"  ⚠️  Поврежденных: {stats['corrupted_count']}")
        
        # Предупреждения и проблемы
        if self.results['warnings']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("7. ПРЕДУПРЕЖДЕНИЯ")
            report_lines.append("=" * 80)
            for warning in self.results['warnings']:
                report_lines.append(f"\n  [{warning['severity'].upper()}] {warning['message']}")
        
        if self.results['issues']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("8. ПРОБЛЕМЫ И ОШИБКИ")
            report_lines.append("=" * 80)
            for issue in self.results['issues'][:10]:  # Показываем первые 10
                report_lines.append(f"\n  Тип: {issue['type']}")
                report_lines.append(f"  Файл: {issue['file']}")
                report_lines.append(f"  Ошибка: {issue['error']}")
            if len(self.results['issues']) > 10:
                report_lines.append(f"\n  ... и еще {len(self.results['issues']) - 10} проблем")
        
        # Рекомендации
        report_lines.append("\n" + "=" * 80)
        report_lines.append("9. РЕКОМЕНДАЦИИ")
        report_lines.append("=" * 80)
        
        if 'statistics' in self.results and 'class_balance' in self.results['statistics']:
            if self.results['statistics']['class_balance']['imbalance_ratio'] > 5:
                report_lines.append("\n  ⚠️  Высокий дисбаланс классов обнаружен!")
                report_lines.append("     Рекомендуется:")
                report_lines.append("     - Использовать Weighted Loss")
                report_lines.append("     - Применить SMOTE oversampling для minority классов")
                report_lines.append("     - Использовать Focal Loss")
        
        if self.results['statistics'].get('corrupted_images', 0) > 0:
            report_lines.append("\n  ⚠️  Обнаружены поврежденные изображения!")
            report_lines.append("     Рекомендуется проверить и удалить поврежденные файлы")
        
        if 'statistics' in self.results and 'width' in self.results['statistics']:
            width_std = self.results['statistics']['width']['std']
            if width_std > 200:
                report_lines.append("\n  ⚠️  Большое разнообразие размеров изображений!")
                report_lines.append("     Рекомендуется:")
                report_lines.append("     - Применить resize/rescale при предобработке")
                report_lines.append("     - Использовать центральный crop")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("КОНЕЦ ОТЧЕТА")
        report_lines.append("=" * 80)
        
        # Сохранение отчета
        report_text = "\n".join(report_lines)
        report_path = output_path / f"dataset_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"   ✅ Отчет сохранен: {report_path}")
        
        # Сохранение JSON
        json_path = output_path / f"dataset_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ JSON данные сохранены: {json_path}")
        
        # Вывод отчета в консоль
        print("\n" + "=" * 80)
        print("КРАТКАЯ СВОДКА")
        print("=" * 80)
        print(report_text)
        
        return report_path
    
    def run_full_analysis(self, output_dir='results'):
        """Запуск полного анализа"""
        print("\n" + "=" * 80)
        print("АНАЛИЗ КАЧЕСТВА ДАТАСЕТА PLANTVILLAGE")
        print("=" * 80)
        print(f"\nПуть к датасету: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            print(f"❌ Ошибка: путь {self.dataset_path} не существует!")
            return
        
        # Выполняем все этапы анализа
        self.analyze_structure()
        self.analyze_images()
        self.analyze_class_balance()
        self.create_visualizations(output_dir)
        report_path = self.generate_report(output_dir)
        
        print("\n" + "=" * 80)
        print("✅ АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 80)
        print(f"Отчет сохранен: {report_path}")
        
        return self.results

if __name__ == "__main__":
    # Путь к датасету
    dataset_path = "/home/ann/Загрузки/archive(1)/plantvillage/PlantVillage/"
    
    # Создаем анализатор
    analyzer = DatasetQualityAnalyzer(dataset_path)
    
    # Запускаем полный анализ
    results = analyzer.run_full_analysis(output_dir='results')

