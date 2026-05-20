import { Network } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

export function GraphEmptyState() {
  const { t } = useLanguage();

  return (
    <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-500">
      <div className="text-center">
        <Network className="w-16 h-16 mx-auto mb-4 opacity-20" />
        <p>{t('graph.clickToView')}</p>
        <p>{t('graph.toViewGraph')}</p>
      </div>
    </div>
  );
}
